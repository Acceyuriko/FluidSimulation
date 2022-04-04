using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Transforms;
using UnityEngine;

public partial class FluidSPHByGPUSystem : SystemBase
{
    private EntityQuery m_ParticleQuery;
    private EntityQuery m_BoundaryQuery;

    protected override void OnCreate()
    {
        m_ParticleQuery = GetEntityQuery(FluidSPHUtils.ParticleQueryComponentTypes);
        m_BoundaryQuery = GetEntityQuery(FluidSPHUtils.BoundaryQueryComponentTypes);
        RequireForUpdate(m_ParticleQuery);
        RequireSingletonForUpdate<SimulationSettings>();
    }

    protected override void OnUpdate()
    {
        var settings = GetSingleton<SimulationSettings>();
        if (!SystemInfo.supportsComputeShaders || settings.UseGPU)
        {
            Enabled = false;
            return;
        }
        World.GetOrCreateSystem<FixedStepSimulationSystemGroup>().Timestep = 1f / settings.FPS;
    }

    protected override void OnDestroy()
    {

    }
}
