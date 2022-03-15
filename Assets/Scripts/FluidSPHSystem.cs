using Unity.Entities;
using Unity.Jobs;
using Unity.Transforms;
using Unity.Physics;

[UpdateInGroup(typeof(SimulationSystemGroup))]
[UpdateBefore(typeof(FixedStepSimulationSystemGroup))]
public class FluidSPHSystem : SystemBase
{
    public BeginFixedStepSimulationEntityCommandBufferSystem m_BeginFixedStepECB;

    protected override void OnCreate()
    {
        m_BeginFixedStepECB = World.GetOrCreateSystem<BeginFixedStepSimulationEntityCommandBufferSystem>();
    }

    protected override void OnUpdate()
    {
        var commandBuffer = m_BeginFixedStepECB.CreateCommandBuffer();

        Entities.ForEach((ref Translation translation, in Rotation rotation) => {
        }).Schedule();
    }

    protected override void OnDestroy()
    {
        base.OnDestroy();
    }
}
