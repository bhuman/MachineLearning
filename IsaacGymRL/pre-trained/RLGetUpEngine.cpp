/**
 * @file RLGetUpEngine.h
 *
 * @Author Philip Reichenberg
 */

#include "RLGetUpEngine.h"
#include "Debugging/Plot.h"
#include "Math/Rotation.h"
#include "Platform/SystemCall.h"
#include "Tools/Motion/MotionUtilities.h"
#include "Tools/Modeling/BallPhysics.h"
#include <filesystem>

MAKE_MODULE(RLGetUpEngine);

RLGetUpEngine::RLGetUpEngine():
  policy(&Global::getAsmjitRuntime())
{
  compile(false);
  // https://github.com/BoosterRobotics/booster_gym/blob/main/deploy/configs/T1.yaml#L19
  offset.angles[Joints::lHipPitch] = -0.2f;
  offset.angles[Joints::lKneePitch] = 0.4f;
  offset.angles[Joints::lAnklePitch] = -0.25f;
  offset.angles[Joints::rHipPitch] = -0.2f;
  offset.angles[Joints::rKneePitch] = 0.4f;
  offset.angles[Joints::rAnklePitch] = -0.25f;

  std::vector<Joints::Joint> upperBodyJoints = { Joints::headYaw,
                                                 Joints::headPitch,
                                                 Joints::lShoulderPitch,
                                                 Joints::lShoulderRoll,
                                                 Joints::lElbowYaw,
                                                 Joints::lElbowRoll,
                                                 Joints::rShoulderPitch,
                                                 Joints::rShoulderRoll,
                                                 Joints::rElbowYaw,
                                                 Joints::rElbowRoll,
                                               };

  std::vector<Joints::Joint> lowerBodyJoints = { Joints::lHipPitch,
                                                 Joints::lHipRoll,
                                                 Joints::lHipYaw,
                                                 Joints::lKneePitch,
                                                 Joints::lAnklePitch,
                                                 Joints::lAnkleRoll,
                                                 Joints::rHipPitch,
                                                 Joints::rHipRoll,
                                                 Joints::rHipYaw,
                                                 Joints::rKneePitch,
                                                 Joints::rAnklePitch,
                                                 Joints::rAnkleRoll
                                               };

  boosterWaistJoints = upperBodyJoints;
  boosterJoints = upperBodyJoints;

  boosterWaistJoints.push_back(Joints::waistYaw);

  boosterWaistJoints.insert(boosterWaistJoints.end(), lowerBodyJoints.begin(), lowerBodyJoints.end());
  boosterJoints.insert(boosterJoints.end(), lowerBodyJoints.begin(), lowerBodyJoints.end());

  recoveryPose.angles.fill(0);
  recoveryPose.angles[Joints::lShoulderRoll] = -80_deg;
  recoveryPose.angles[Joints::rShoulderRoll] = 80_deg;

  fallAngles.angles.fill(0);
  fallAngles.angles[Joints::headPitch] = fallAngles.angles[Joints::headYaw] = 0_deg;
  fallAngles.angles[Joints::lShoulderPitch] = fallAngles.angles[Joints::rShoulderPitch] = 0_deg;
  fallAngles.angles[Joints::lShoulderRoll] = -83_deg;
  fallAngles.angles[Joints::lElbowYaw] = 90_deg;
  fallAngles.angles[Joints::lElbowRoll] = fallAngles.angles[Joints::rElbowRoll] = 0_deg;
  fallAngles.angles[Joints::rShoulderRoll] = 83_deg;
  fallAngles.angles[Joints::rElbowYaw] = 90_deg;
}

void RLGetUpEngine::compile(bool output)
{
  if(output)
  {
    if(!std::filesystem::exists(modelPath + policyName))
    {
      OUTPUT_ERROR("File " << modelPath << policyName << " does not exist");
      return;
    }
  }
  else
    ASSERT(std::filesystem::exists(modelPath + policyName));

  policy.compile(Model(modelPath + policyName));
  ASSERT(policy.valid());

  ASSERT(policy.numOfInputs() == 1);
  ASSERT(policy.input(0).rank() == 1);
  ASSERT(policy.input(0).dims(0) == numInput);

  ASSERT(policy.numOfOutputs() == 1);
  ASSERT(policy.output(0).rank() == 1);
  ASSERT(policy.output(0).dims(0) == numOutput);
}

void RLGetUpEngine::update(GetUpGenerator& theGetUpGenerator)
{
  bool calcVelocity = true;
  if(lastFrameInfo == 0 || lastFrameInfo > theFrameInfo.time)
  {
    lastFrameInfo = theFrameInfo.time;
    calcVelocity = false;
  }
  const float numberFrames = std::max(1.f, std::floor((theFrameInfo.time - lastFrameInfo) / 2.f));
  if(numberFrames > 2.5f) // Booster robots sometimes have longer data drops. In that case it is better to use boosters velocity value
    calcVelocity = false;
  FOREACH_ENUM(Joints::Joint, joint)
  {
    lastMeasurement.velocity[joint] = !calcVelocity ? static_cast<float>(theJointAngles.velocity[joint]) : (theJointAngles.angles[joint] - lastMeasurement.angles[joint]) * 500.f / numberFrames;
  }
  lastFrameInfo = theFrameInfo.time;
  lastMeasurement.angles = theJointAngles.angles;

  theGetUpGenerator.createPhase = [this](const MotionPhase&)->std::unique_ptr<MotionPhase>
  {
    return std::make_unique<RLGetUpPhase>(*this);
  };
}

RLGetUpPhase::RLGetUpPhase(RLGetUpEngine& engine):
  MotionPhase(MotionPhase::getUp),
  engine(engine)
{
  setUpRecovery();
}

void RLGetUpPhase::executePolicy(JointAngles& target)
{
  // Policy learned with 50 hz -> every 20 ms inference
  if(engine.theFrameInfo.getTimeSince(lastInference) < 20)
    return;

  if(maxStandUpTime == 0.f)
  {
    ASSERT(maxStandUpTime > 0.f);
    target.angles = engine.theJointAngles.angles;
    return;
  }

  JointAngles clippedJointAngles = engine.theJointAngles;
  FOREACH_ENUM(Joints::Joint, joint)
    clippedJointAngles.angles[joint] = engine.theJointLimits.limits[joint].limit(clippedJointAngles.angles[joint]);

  float* input = engine.policy.input(0).data();

  const Vector3f gravity = engine.theInertialData.orientation3D.inverse() * Vector3f(0.f, 0.f, -1.f);

  // gravity
  *input++ = gravity.x();
  *input++ = gravity.y();
  *input++ = gravity.z();

  // angular momentum
  *input++ = engine.theInertialData.gyro.x();
  *input++ = engine.theInertialData.gyro.y();
  *input++ = engine.theInertialData.gyro.z();

  // phase
  *input++ = Rangef::ZeroOneRange().limit(executedTime / maxStandUpTime); // value between 0 (start) and 1 (end)

  // joint sequence
  std::vector<Joints::Joint> jointList = getBoosterLegJointSequence();

  // Measurements
  for(Joints::Joint j : jointList)
    *input++ = clippedJointAngles.angles[j] - engine.offset.angles[j];

  // Velocities
  for(Joints::Joint j : jointList)
    *input++ = engine.lastMeasurement.velocity[j] * 0.1f;

  // Last Requests
  for(Joints::Joint j : jointList)
    *input++ = engine.theJointRequest.angles[j] - engine.offset.angles[j];

  // Run network.
  STOPWATCH("module:RLGetUpEngine:apply")
    engine.policy.apply();

  // Get next request
  const float* output = engine.policy.output(0).data();
  for(Joints::Joint j : jointList)
    target.angles[j] = engine.clipActions.limit(*output++) + engine.offset.angles[j];

  lastInference = engine.theFrameInfo.time;
}

void RLGetUpPhase::update()
{
  if(shouldBreakUp())
  {
    state = State::breakUp;
    startTime = engine.theFrameInfo.time;
    tryCounter += SystemCall::getMode() == SystemCall::simulatedRobot ? 0 : 1;
  }

  if(tryCounter >= engine.maxTryCounter)
    state = State::helpMe;

  switch(state)
  {
    case State::recovery:
    {
      ASSERT(recoverMotion);

      if(recoverMotion && engine.theFrameInfo.getTimeSince(startTime) >= (*recoverMotion)[recoverMotionIndex].duration)
      {
        recoverMotionIndex++;
        if(recoverMotionIndex < recoverMotion->size())  // keep executing recover motion
        {
          startAngles.angles = engine.theJointRequest.angles;
        }
        else if(engine.theGameState.isPenalized() && engine.theGameState.gameControllerActive && SystemCall::getMode() != SystemCall::simulatedRobot)
          state = State::helpMe;
        else // Start get up
        {
          state = State::standUp;
          isFront = engine.theInertialData.angle.y() > 0;
          maxStandUpTime = isFront ? engine.frontInfo.back().executionTime : engine.backInfo.back().executionTime;
        }
        executedTime = 0.f;
        startTime = engine.theFrameInfo.time;
      }
      break;
    }
    case State::breakUp:
    {
      if(engine.theFrameInfo.getTimeSince(startTime) >= engine.breakUpTime  // Waited long enough after break up
         && tryCounter < engine.maxTryCounter) // We still have at least one try left
      {
        if(!engine.theGameState.stopped) // The game is currently NOT stopped
          setUpRecovery();
        else if(engine.theFrameInfo.getTimeSince(lastStopSoundTimestamp) > engine.stopSoundTime)
        {
          lastStopSoundTimestamp = engine.theFrameInfo.time;
          SystemCall::say("Stand Up Paused", true);
        }
      }
      break;
    }
    case State::standUp:
    {
      executedTime = engine.theFrameInfo.getTimeSince(startTime) * engine.speedFactor;
      if((executedTime / maxStandUpTime >= 1.f
          || (executedTime / maxStandUpTime >= engine.earliestDoneTime
              && -engine.theTorsoMatrix.translation.z() > engine.minStandHeightWhenDone))
         && (engine.theFallDownState.state == FallDownState::upright || engine.theFallDownState.state == FallDownState::staggering))
        state = State::done;
      break;
    }
    case State::helpMe:
    {
      if(!engine.theGameState.isPenalized() && tryCounter < engine.maxTryCounter)
        setUpRecovery();
      else if(engine.theFrameInfo.getTimeSince(lastHelpMeSound) > engine.helpMeSoundTimeWindow)
      {
        lastHelpMeSound = engine.theFrameInfo.time;
        SystemCall::say("Help me");
        SystemCall::playSound("mimimi.wav");
      }
      break;
    }
  }
}

bool RLGetUpPhase::isDone(const MotionRequest& request) const
{
  return request.motion == MotionRequest::playDead
         || request.motion == MotionRequest::prepare
         || state == State::done;
}

void RLGetUpPhase::calcJoints(const MotionRequest&, JointRequest& jointRequest, Pose2f& odometryOffset, MotionInfo& motionInfo)
{
  switch(state)
  {
    case State::recovery:
      doRecovery(nextRequest);
      break;
    case State::standUp:
      executePolicy(nextRequest);
      break;
    case State::breakUp:
    case State::helpMe:
      doBreakUp(nextRequest);
      break;
  }

  jointRequest = nextRequest;

  // else set head, and keep everything default, so walk can overwrite it

  motionInfo.isMotionStable = false;
  motionInfo.isWalkPhaseInWalkKick = false;
  motionInfo.speed = Pose2f();

  odometryOffset = Pose2f(0_deg, 0.f, 0.f);
}

void RLGetUpPhase::doRecovery(JointRequest& request)
{
  ASSERT(recoverMotion);
  const float ratio = std::min(engine.theFrameInfo.getTimeSince(startTime) / (*recoverMotion)[recoverMotionIndex].duration, 1.f);
  FOREACH_ENUM(Joints::Joint, joint)
    request.angles[joint] = startAngles.angles[joint] * (1.f - ratio) + (*recoverMotion)[recoverMotionIndex].positions[joint] * ratio;
  request.stiffnessData.stiffnesses.fill(100);
}

void RLGetUpPhase::getRefTorso(Rangea& torsoXRange, Rangea& torsoYRange)
{
  ASSERT(state != State::recovery);
  ASSERT(state != State::breakUp);
  ASSERT(engine.frontInfo.size() > 1);
  ASSERT(engine.backInfo.size() > 1);
  const float executedRatio = executedTime / engine.recoveryTime;
  const auto& refInfo = isFront ? engine.frontInfo : engine.backInfo;

  std::size_t startIndex = 0;
  std::size_t nextIndex = 0;
  for(std::size_t index = 0; index < refInfo.size(); index++)
  {
    const float nextMaxRatio = refInfo[index].executionTime / engine.recoveryTime;
    if(nextMaxRatio < executedRatio)
      startIndex = index;
    else
    {
      nextIndex = index;
      break;
    }
  }

  const float refStartTime = refInfo[startIndex].executionTime;
  const float refNextTime = refInfo[nextIndex].executionTime;
  const float torsoRatio = Rangef::ZeroOneRange().limit((executedTime - refStartTime) / (refNextTime - refStartTime));
  torsoXRange.min = refInfo[startIndex].torsoXRange.min * (1.f - torsoRatio) + refInfo[nextIndex].torsoXRange.min * torsoRatio;
  torsoXRange.max = refInfo[startIndex].torsoXRange.max * (1.f - torsoRatio) + refInfo[nextIndex].torsoXRange.max * torsoRatio;
  torsoYRange.min = refInfo[startIndex].torsoYRange.min * (1.f - torsoRatio) + refInfo[nextIndex].torsoYRange.min * torsoRatio;
  torsoYRange.max = refInfo[startIndex].torsoYRange.max * (1.f - torsoRatio) + refInfo[nextIndex].torsoYRange.max * torsoRatio;
}

bool RLGetUpPhase::shouldBreakUp()
{
  if(state == State::standUp && executedTime > (isFront ? engine.frontInfo : engine.backInfo)[0].executionTime)
  {
    Rangea torsoXRange;
    Rangea torsoYRange;
    getRefTorso(torsoXRange, torsoYRange);
    return !torsoXRange.isInside(engine.theInertialData.angle.x()) || !torsoYRange.isInside(engine.theInertialData.angle.y());
  }
  return false;
}

void RLGetUpPhase::doBreakUp(JointRequest& request)
{
  const Angle jointSpeed = engine.breakUpJointSpeed * Global::getSettings().motionCycleTime;
  const Rangea speedLimit(-jointSpeed, jointSpeed);
  JointAngles targetAngles = engine.fallAngles;
  targetAngles.angles[Joints::headPitch] = (engine.theInertialData.angle.y() > 0.f ? -1.f : 1.f) * engine.breakUpHeadAngle;
  request.stiffnessData.stiffnesses.fill(50);
  request.angles = engine.theJointRequest.angles;
  FOREACH_ENUM(Joints::Joint, joint)
  {
    if(joint > Joints::firstLegJoint)
      request.angles[joint] = engine.theJointAngles.angles[joint];
    const Rangea positionLimit(-engine.maxPositionDifference + engine.theJointAngles.angles[joint], engine.maxPositionDifference + engine.theJointAngles.angles[joint]);
    const Angle maxTargetPosition = positionLimit.limit(targetAngles.angles[joint]);
    request.angles[joint] += speedLimit.limit(maxTargetPosition - engine.theJointRequest.angles[joint]);
  }
}

std::unique_ptr<MotionPhase> RLGetUpPhase::createNextPhase(const MotionPhase& nextPhase) const
{
  if(nextPhase.type == MotionPhase::prepare || nextPhase.type == MotionPhase::playDead)
    return std::unique_ptr<MotionPhase>();
  // Force walk afterwards
  return engine.theWalkGenerator.createPhase(Pose2f(0.f, 0.01f, 0.f), *this, 0.f);
}

std::vector<Joints::Joint> RLGetUpPhase::getBoosterLegJointSequence()
{
  if(Global::getSettings().robotType != Settings::nao)
  {
    if(engine.useWaist)
      return engine.boosterWaistJoints;
    return engine.boosterJoints;
  }
  else
  {
    FAIL("RLGetUpEngine not implemented for NAO!");
    return engine.boosterJoints;
  }
}

void RLGetUpPhase::setUpRecovery()
{
  startTime = engine.theFrameInfo.time;
  startAngles = engine.theJointAngles;

  state = State::recovery;
  if(std::abs(engine.theInertialData.angle.x()) >= 40_deg && engine.theInertialData.angle.y() < 30_deg)
    recoverMotion = &engine.recoverBack;
  else if(std::abs(engine.theInertialData.angle.x()) >= 40_deg)
    recoverMotion = &engine.recoverFront;
  else
    recoverMotion = &engine.recoverNormal;
  recoverMotionIndex = 0;
  executedTime = 0.f;
}
