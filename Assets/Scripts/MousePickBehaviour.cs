using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Systems;
using Unity.Transforms;
using UnityEngine;
using UnityEngine.Assertions;
using static Unity.Physics.Math;

namespace Unity.Physics.Extensions
{
    // A mouse pick collector which stores every hit. Based off the ClosestHitCollector
    [BurstCompile]
    public struct MousePickCollector : ICollector<RaycastHit>
    {
        public bool IgnoreTriggers;
        public NativeArray<RigidBody> Bodies;
        public int NumDynamicBodies;

        public bool EarlyOutOnFirstHit => false;
        public float MaxFraction { get; private set; }
        public int NumHits { get; private set; }

        private RaycastHit m_ClosestHit;
        public RaycastHit Hit => m_ClosestHit;

        public MousePickCollector(float maxFraction, NativeArray<RigidBody> rigidBodies, int numDynamicBodies)
        {
            m_ClosestHit = default(RaycastHit);
            MaxFraction = maxFraction;
            NumHits = 0;
            IgnoreTriggers = true;
            Bodies = rigidBodies;
            NumDynamicBodies = numDynamicBodies;
        }

        #region ICollector

        public bool AddHit(RaycastHit hit)
        {
            Assert.IsTrue(hit.Fraction < MaxFraction);

            var isAcceptable = (hit.RigidBodyIndex >= 0) && (hit.RigidBodyIndex < NumDynamicBodies);
            if (IgnoreTriggers)
            {
                isAcceptable = isAcceptable && hit.Material.CollisionResponse != CollisionResponsePolicy.RaiseTriggerEvents;
            }

            if (!isAcceptable)
            {
                return false;
            }

            MaxFraction = hit.Fraction;
            m_ClosestHit = hit;
            NumHits = 1;
            return true;
        }

        #endregion
    }

    public struct MousePick : IComponentData
    {
        public int IgnoreTriggers;
    }

    public class MousePickBehaviour : MonoBehaviour, IConvertGameObjectToEntity
    {
        public bool IgnoreTriggers = true;

        void IConvertGameObjectToEntity.Convert(Entity entity, EntityManager dstManager, GameObjectConversionSystem conversionSystem)
        {
            dstManager.AddComponentData(entity, new MousePick()
            {
                IgnoreTriggers = IgnoreTriggers ? 1 : 0,
            });
        }
    }

    // Attaches a virtual spring to the picked entity
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    public partial class MousePickSystem : SystemBase
    {
        const float k_MaxDistance = 100.0f;

        EntityQuery m_MouseGroup;
        BuildPhysicsWorld m_BuildPhysicsWorldSystem;

        public NativeArray<SpringData> SpringDatas;
        public JobHandle? PickJobHandle;

        public struct SpringData
        {
            public Entity Entity;
            public int Dragging; // bool isn't blittable
            public float3 PointOnBody;
            public float MouseDepth;
        }


        [BurstCompile]
        struct Pick : IJob
        {
            [ReadOnly] public CollisionWorld CollisionWorld;
            [ReadOnly] public int NumDynamicBodies;
            public NativeArray<SpringData> SpringData;
            public RaycastInput RayInput;
            public float Near;
            public float3 Forward;
            [ReadOnly] public bool IgnoreTriggers;

            public void Execute()
            {
                var mousePickCollector = new MousePickCollector(1.0f, CollisionWorld.Bodies, NumDynamicBodies);
                mousePickCollector.IgnoreTriggers = IgnoreTriggers;

                CollisionWorld.CastRay(RayInput, ref mousePickCollector);
                if (mousePickCollector.MaxFraction < 1.0f)
                {
                    float fraction = mousePickCollector.Hit.Fraction;
                    RigidBody hitBody = CollisionWorld.Bodies[mousePickCollector.Hit.RigidBodyIndex];

                    MTransform bodyFromWorld = Inverse(new MTransform(hitBody.WorldFromBody));
                    float3 pointOnBody = Mul(bodyFromWorld, mousePickCollector.Hit.Position);

                    SpringData[0] = new SpringData
                    {
                        Entity = hitBody.Entity,
                        Dragging = 1,
                        PointOnBody = pointOnBody,
                        MouseDepth = Near + math.dot(math.normalize(RayInput.End - RayInput.Start), Forward) * fraction * k_MaxDistance,
                    };
                }
                else
                {
                    SpringData[0] = new SpringData
                    {
                        Dragging = 0
                    };
                }
            }
        }

        public MousePickSystem()
        {
            SpringDatas = new NativeArray<SpringData>(1, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            SpringDatas[0] = new SpringData();
        }

        protected override void OnCreate()
        {
            m_BuildPhysicsWorldSystem = World.GetOrCreateSystem<BuildPhysicsWorld>();
            m_MouseGroup = GetEntityQuery(new EntityQueryDesc
            {
                All = new ComponentType[] { typeof(MousePick) }
            });
        }

        protected override void OnDestroy()
        {
            SpringDatas.Dispose();
        }

        protected override void OnStartRunning()
        {
            m_BuildPhysicsWorldSystem.RegisterPhysicsRuntimeSystemReadOnly();
        }

        protected override void OnUpdate()
        {
            if (m_MouseGroup.CalculateEntityCount() == 0)
            {
                return;
            }

            var handle = Dependency;

            if (Input.GetMouseButtonDown(0) && (Camera.main != null))
            {
                Vector2 mousePosition = Input.mousePosition;
                UnityEngine.Ray unityRay = Camera.main.ScreenPointToRay(mousePosition);

                var mice = m_MouseGroup.ToComponentDataArray<MousePick>(Allocator.TempJob);
                var IgnoreTriggers = mice[0].IgnoreTriggers != 0;
                mice.Dispose();

                // Schedule picking job, after the collision world has been built
                handle = new Pick
                {
                    CollisionWorld = m_BuildPhysicsWorldSystem.PhysicsWorld.CollisionWorld,
                    NumDynamicBodies = m_BuildPhysicsWorldSystem.PhysicsWorld.NumDynamicBodies,
                    SpringData = SpringDatas,
                    RayInput = new RaycastInput
                    {
                        Start = unityRay.origin,
                        End = unityRay.origin + unityRay.direction * k_MaxDistance,
                        Filter = new CollisionFilter
                        {
                            BelongsTo = ~0u,
                            CollidesWith = (uint)EPhysicsCategoryNames.ParticleCollider
                        },
                    },
                    Near = Camera.main.nearClipPlane,
                    Forward = Camera.main.transform.forward,
                    IgnoreTriggers = IgnoreTriggers,
                }.Schedule(handle);

                PickJobHandle = handle;

                handle.Complete(); // TODO.ma figure out how to do this properly...we need a way to make physics sync wait for
                // any user jobs that touch the component data, maybe a JobHandle LastUserJob or something that the user has to set
            }

            if (Input.GetMouseButtonUp(0))
            {
                SpringDatas[0] = new SpringData();
            }

            Dependency = handle;
        }
    }

    // Applies any mouse spring as a change in velocity on the entity's motion component
    [UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
    [UpdateBefore(typeof(BuildPhysicsWorld))]
    public partial class MouseSpringSystem : SystemBase
    {
        EntityQuery m_MouseGroup;
        MousePickSystem m_PickSystem;

        protected override void OnCreate()
        {
            m_PickSystem = World.GetOrCreateSystem<MousePickSystem>();
            m_MouseGroup = GetEntityQuery(new EntityQueryDesc
            {
                All = new ComponentType[] { typeof(MousePick) }
            });
        }

        protected override void OnUpdate()
        {
            if (m_MouseGroup.CalculateEntityCount() == 0)
            {
                return;
            }

            ComponentDataFromEntity<Translation> Positions = GetComponentDataFromEntity<Translation>(true);
            ComponentDataFromEntity<Rotation> Rotations = GetComponentDataFromEntity<Rotation>(true);
            ComponentDataFromEntity<PhysicsVelocity> Velocities = GetComponentDataFromEntity<PhysicsVelocity>();
            ComponentDataFromEntity<PhysicsMass> Masses = GetComponentDataFromEntity<PhysicsMass>(true);

            // If there's a pick job, wait for it to finish
            if (m_PickSystem.PickJobHandle != null)
            {
                JobHandle.CombineDependencies(Dependency, m_PickSystem.PickJobHandle.Value).Complete();
            }

            // If there's a picked entity, drag it
            MousePickSystem.SpringData springData = m_PickSystem.SpringDatas[0];
            if (springData.Dragging != 0)
            {
                Entity entity = m_PickSystem.SpringDatas[0].Entity;
                if (!EntityManager.HasComponent<PhysicsMass>(entity))
                {
                    return;
                }

                PhysicsMass massComponent = Masses[entity];
                PhysicsVelocity velocityComponent = Velocities[entity];

                if (massComponent.InverseMass == 0)
                {
                    return;
                }

                var worldFromBody = new MTransform(Rotations[entity].Value, Positions[entity].Value);

                // Body to motion transform
                var bodyFromMotion = new MTransform(Masses[entity].InertiaOrientation, Masses[entity].CenterOfMass);
                MTransform worldFromMotion = Mul(worldFromBody, bodyFromMotion);

                // Damp the current velocity
                const float gain = 0.95f;
                velocityComponent.Linear *= gain;
                velocityComponent.Angular *= gain;

                // Get the body and mouse points in world space
                float3 pointBodyWs = Mul(worldFromBody, springData.PointOnBody);
                float3 pointSpringWs = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, springData.MouseDepth));

                // Calculate the required change in velocity
                float3 pointBodyLs = Mul(Inverse(bodyFromMotion), springData.PointOnBody);
                float3 deltaVelocity;
                {
                    float3 pointDiff = pointBodyWs - pointSpringWs;
                    float3 relativeVelocityInWorld = velocityComponent.Linear + math.mul(worldFromMotion.Rotation, math.cross(velocityComponent.Angular, pointBodyLs));

                    const float elasticity = 0.1f;
                    const float damping = 0.5f;
                    deltaVelocity = -pointDiff * (elasticity / Time.DeltaTime) - damping * relativeVelocityInWorld;
                }

                // Build effective mass matrix in world space
                // TODO how are bodies with inf inertia and finite mass represented
                // TODO the aggressive damping is hiding something wrong in this code if dragging non-uniform shapes
                float3x3 effectiveMassMatrix;
                {
                    float3 arm = pointBodyWs - worldFromMotion.Translation;
                    var skew = new float3x3(
                        new float3(0.0f, arm.z, -arm.y),
                        new float3(-arm.z, 0.0f, arm.x),
                        new float3(arm.y, -arm.x, 0.0f)
                    );

                    // world space inertia = worldFromMotion * inertiaInMotionSpace * motionFromWorld
                    var invInertiaWs = new float3x3(
                        massComponent.InverseInertia.x * worldFromMotion.Rotation.c0,
                        massComponent.InverseInertia.y * worldFromMotion.Rotation.c1,
                        massComponent.InverseInertia.z * worldFromMotion.Rotation.c2
                    );
                    invInertiaWs = math.mul(invInertiaWs, math.transpose(worldFromMotion.Rotation));

                    float3x3 invEffMassMatrix = math.mul(math.mul(skew, invInertiaWs), skew);
                    invEffMassMatrix.c0 = new float3(massComponent.InverseMass, 0.0f, 0.0f) - invEffMassMatrix.c0;
                    invEffMassMatrix.c1 = new float3(0.0f, massComponent.InverseMass, 0.0f) - invEffMassMatrix.c1;
                    invEffMassMatrix.c2 = new float3(0.0f, 0.0f, massComponent.InverseMass) - invEffMassMatrix.c2;

                    effectiveMassMatrix = math.inverse(invEffMassMatrix);
                }

                // Calculate impulse to cause the desired change in velocity
                float3 impulse = math.mul(effectiveMassMatrix, deltaVelocity);

                // Clip the impulse
                const float maxAcceleration = 250.0f;
                float maxImpulse = math.rcp(massComponent.InverseMass) * Time.DeltaTime * maxAcceleration;
                impulse *= math.min(1.0f, math.sqrt((maxImpulse * maxImpulse) / math.lengthsq(impulse)));

                // Apply the impulse
                {
                    velocityComponent.Linear += impulse * massComponent.InverseMass;

                    float3 impulseLs = math.mul(math.transpose(worldFromMotion.Rotation), impulse);
                    float3 angularImpulseLs = math.cross(pointBodyLs, impulseLs);
                    velocityComponent.Angular += angularImpulseLs * massComponent.InverseInertia;
                }

                // Write back velocity
                Velocities[entity] = velocityComponent;
            }
        }
    }
}
