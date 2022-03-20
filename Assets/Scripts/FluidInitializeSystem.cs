using Unity.Entities;
using Unity.Mathematics;
using Unity.Rendering;
using Unity.Transforms;
using UnityEngine;

[UpdateInGroup(typeof(InitializationSystemGroup))]
[UpdateBefore(typeof(EndInitializationEntityCommandBufferSystem))]
public class FluidInitializeSystem : SystemBase
{
    private EndInitializationEntityCommandBufferSystem m_EndInitializationECB;

    protected override void OnCreate()
    {
        m_EndInitializationECB = World.GetOrCreateSystem<EndInitializationEntityCommandBufferSystem>();
        RequireForUpdate(GetEntityQuery(new EntityQueryDesc
        {
            Any = new ComponentType[] { typeof(FluidComponent), typeof(BoundaryComponent) }
        }));
    }

    protected override void OnUpdate()
    {
        var commandBuffer = m_EndInitializationECB.CreateCommandBuffer();

        Entities
            .WithoutBurst()
            .WithAll<FluidComponent>()
            .ForEach((Entity entity, in LocalToWorld localToWorld, in RenderBounds bounds, in FluidComponent fluid) =>
            {
                Debug.Log("Create fluid particles");
                CreateParticles(
                    localToWorld.Value,
                    bounds.Value,
                    EntityManager.GetComponentData<FluidParticleComponent>(fluid.prefab).radius,
                    commandBuffer,
                    fluid.prefab
                );
                commandBuffer.DestroyEntity(entity);
            })
            .Run();

        Entities
            .WithoutBurst()
            .WithAll<BoundaryComponent>()
            .ForEach((Entity entity, in LocalToWorld localToWorld, in RenderBounds bounds, in BoundaryComponent boundary) =>
            {
                Debug.Log("Create boundary particles");
                CreateParticles(
                    localToWorld.Value,
                    bounds.Value,
                    EntityManager.GetComponentData<BoundaryParticleComponent>(boundary.prefab).radius,
                    commandBuffer,
                    boundary.prefab
                );
                commandBuffer.RemoveComponent<BoundaryComponent>(entity);
            })
            .Run();

        m_EndInitializationECB.AddJobHandleForProducer(Dependency);
    }

    private static void CreateParticles(Matrix4x4 l2w, AABB bounds, float radius, EntityCommandBuffer cb, Entity prefab)
    {
        var xRadius = radius / l2w.MultiplyVector(Vector3.right).magnitude;
        var yRadius = radius / l2w.MultiplyVector(Vector3.up).magnitude;
        var zRadius = radius / l2w.MultiplyVector(Vector3.forward).magnitude;

        int numX = (int)((bounds.Size.x + xRadius) / (2 * xRadius));
        int numY = (int)((bounds.Size.y + yRadius) / (2 * yRadius));
        int numZ = (int)((bounds.Size.z + zRadius) / (2 * zRadius));

        for (int z = 0; z < numZ; z++)
        {
            for (int y = 0; y < numY; y++)
            {
                for (int x = 0; x < numX; x++)
                {
                    var e = cb.Instantiate(prefab);
                    cb.SetComponent(e, new Translation
                    {
                        Value = l2w.MultiplyPoint3x4(
                            new Vector3(
                                x * 2 * xRadius + bounds.Min.x + xRadius,
                                y * 2 * yRadius + bounds.Min.y + yRadius,
                                z * 2 * zRadius + bounds.Min.z + zRadius
                            )
                        )
                    });
                }
            }
        }

        Debug.Log($"numX: {numX}, numY: {numY}, numZ: {numZ} Particles number: {numX * numY * numZ}");
    }
}
