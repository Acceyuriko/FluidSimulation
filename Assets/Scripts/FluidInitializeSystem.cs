using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Rendering;
using Unity.Transforms;
using UnityEngine;

[UpdateInGroup(typeof(InitializationSystemGroup))]
[UpdateBefore(typeof(EndInitializationEntityCommandBufferSystem))]
public class FluidInitializeSystem : SystemBase
{
    private struct InitializeTag : IComponentData { }

    private EndInitializationEntityCommandBufferSystem m_EndInitializationECB;
    private Entity m_ParticlePrefab;

    protected override void OnCreate()
    {
        m_EndInitializationECB = World.GetOrCreateSystem<EndInitializationEntityCommandBufferSystem>();

        EntityManager.CreateEntity(typeof(InitializeTag));
        RequireSingletonForUpdate<InitializeTag>();
        RequireSingletonForUpdate<PrefabCollection>();
    }

    protected override void OnUpdate()
    {
        if (m_ParticlePrefab == Entity.Null)
        {
            m_ParticlePrefab = GetSingleton<PrefabCollection>().ParticlePrefab;
            return;
        }

        var commandBuffer = m_EndInitializationECB.CreateCommandBuffer();
        commandBuffer.DestroyEntity(GetSingletonEntity<InitializeTag>());

        var prefab = m_ParticlePrefab;
        Debug.Log("FluidInitializeSystem: " + prefab);

        Entities
            .WithAll<FluidTag>()
            .ForEach((Entity entity, in LocalToWorld localToWorld, in RenderBounds rbounds, in FluidTag fluidTag) =>
            {
                Bounds bounds = new Bounds();
                float4 min = new float4(math.mul(localToWorld.Value, new float4(rbounds.Value.Min, 1)));
                float4 max = new float4(math.mul(localToWorld.Value, new float4(rbounds.Value.Max, 1)));

                min.x += fluidTag.radius;
                min.y += fluidTag.radius;
                min.z += fluidTag.radius;

                max.x -= fluidTag.radius;
                max.y -= fluidTag.radius;
                max.z -= fluidTag.radius;

                bounds.SetMinMax(min.xyz, max.xyz);

                float diameter = fluidTag.radius * 2;
                float spacing = diameter * 0.9f;
                float halfSpacing = spacing * 0.5f;

                int numX = (int)((bounds.size.x + halfSpacing) / spacing);
                int numY = (int)((bounds.size.y + halfSpacing) / spacing);
                int numZ = (int)((bounds.size.z + halfSpacing) / spacing);

                Debug.Log($"Particles number: {numX * numY * numZ}");

                for (int z = 0; z < numZ; z++)
                {
                    for (int y = 0; y < numY; y++)
                    {
                        for (int x = 0; x < numX; x++)
                        {
                            var particle = commandBuffer.Instantiate(prefab);
                            commandBuffer.AddComponent(particle, new Scale { Value = 2 * fluidTag.radius });
                            commandBuffer.SetComponent(particle, new Translation
                            {
                                Value = math.mul(localToWorld.Rotation,
                                    new float3(
                                        spacing * x + bounds.min.x + halfSpacing,
                                        spacing * y + bounds.min.y + halfSpacing,
                                        spacing * z + bounds.min.z + halfSpacing
                                    )
                                )
                            });
                            commandBuffer.SetComponent(particle, new PhysicsCollider
                            {
                                Value = Unity.Physics.SphereCollider.Create(new SphereGeometry { Center = float3.zero, Radius = fluidTag.radius })
                            });
                        }
                    }
                }
                commandBuffer.DestroyEntity(entity);
            })
            .Schedule();

        m_EndInitializationECB.AddJobHandleForProducer(Dependency);
    }
}
