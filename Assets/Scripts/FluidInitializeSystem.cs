using Unity.Entities;
using Unity.Mathematics;
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
            .ForEach((Entity entity, in LocalToWorld localToWorld, in RenderBounds bounds, in FluidTag fluidTag) =>
            {
                var halfX = math.distance(localToWorld.Right, localToWorld.Position);
                var halfY = math.distance(localToWorld.Up, localToWorld.Position);
                var halfZ = math.distance(localToWorld.Forward, localToWorld.Position);
                Debug.Log($"halfX: {halfX}, halfY: {halfY}, halfZ: {halfZ}");
                Debug.Log($"position: {localToWorld.Position}");
                for (int i = 0; i < 1000; i++)
                {
                    var particle = commandBuffer.Instantiate(prefab);
                    var radius = fluidTag.radius;
                    commandBuffer.SetComponent(particle, new Translation
                    {
                        Value = localToWorld.Position + new float3(
                            (i / 16 / 16) * radius,
                            (i % 16) * radius,
                            ((i / 16) % 16) * radius
                        )
                    });
                    commandBuffer.SetComponent(particle, new Rotation{
                        Value = localToWorld.Rotation
                    });
                }
                commandBuffer.DestroyEntity(entity);
            })
            .Schedule();

        m_EndInitializationECB.AddJobHandleForProducer(Dependency);
    }
}
