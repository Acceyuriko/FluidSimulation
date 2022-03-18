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

    protected override void OnCreate()
    {
        m_EndInitializationECB = World.GetOrCreateSystem<EndInitializationEntityCommandBufferSystem>();

        EntityManager.CreateEntity(typeof(InitializeTag));
        RequireSingletonForUpdate<InitializeTag>();
        RequireSingletonForUpdate<PrefabCollection>();
    }

    protected override void OnUpdate()
    {
        var commandBuffer = m_EndInitializationECB.CreateCommandBuffer();
        commandBuffer.DestroyEntity(GetSingletonEntity<InitializeTag>());

        Entities
            .WithoutBurst()
            .WithAll<FluidComponent>()
            .ForEach((Entity entity, in LocalToWorld localToWorld, in RenderBounds rbounds, in FluidComponent fluid) =>
            {
                Bounds bounds = new Bounds();
                float4 min = new float4(math.mul(localToWorld.Value, new float4(rbounds.Value.Min, 1)));
                float4 max = new float4(math.mul(localToWorld.Value, new float4(rbounds.Value.Max, 1)));

                var radius = EntityManager.GetComponentData<FluidParticleComponent>(fluid.prefab).radius;

                min.x += radius;
                min.y += radius;
                min.z += radius;

                max.x -= radius;
                max.y -= radius;
                max.z -= radius;

                bounds.SetMinMax(min.xyz, max.xyz);

                float diameter = radius * 2;
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
                            var particle = commandBuffer.Instantiate(fluid.prefab);
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
                        }
                    }
                }

                commandBuffer.DestroyEntity(entity);
            })
            .Run();

        m_EndInitializationECB.AddJobHandleForProducer(Dependency);
    }
}
