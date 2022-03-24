using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Physics;
using Unity.Transforms;
using UnityEngine;

public class FluidSPHByGPUSystem : SystemBase
{
    private EntityQuery m_ParticleQuery;
    private EntityQuery m_BoundaryQuery;

    protected override void OnCreate()
    {
        if (!SystemInfo.supportsComputeShaders)
        {
            Enabled = false;
            return;
        }

        m_ParticleQuery = GetEntityQuery(new EntityQueryDesc
        {
            All = new ComponentType[] {
                typeof(FluidParticleComponent),
                typeof(Translation),
                typeof(PhysicsVelocity),
                typeof(PhysicsMass),
                typeof(PhysicsCollider),
            }
        });

        m_BoundaryQuery = GetEntityQuery(new EntityQueryDesc
        {
            All = new ComponentType[] {
                typeof(BoundaryParticleComponent),
                typeof(Translation)
            }
        });
    }

    protected override void OnUpdate()
    {
        var positions = m_ParticleQuery.ToComponentDataArrayAsync<Translation>(Allocator.TempJob, out var positionHandle);
        var particles = m_ParticleQuery.ToComponentDataArrayAsync<FluidParticleComponent>(Allocator.TempJob, out var particleHandle);
        var physicsMasses = m_ParticleQuery.ToComponentDataArrayAsync<PhysicsMass>(Allocator.TempJob, out var physicsMassHandle);
        var physicsVelocities = m_ParticleQuery.ToComponentDataArrayAsync<PhysicsVelocity>(Allocator.TempJob, out var physicsVelocityHandle);

        Dependency = JobHandle.CombineDependencies(
            positionHandle,
            particleHandle,
            JobHandle.CombineDependencies(
                physicsMassHandle,
                physicsVelocityHandle
            )
        );

        var boundaryPositions = m_BoundaryQuery.ToComponentDataArrayAsync<Translation>(Allocator.TempJob, out var boundaryPositionHandle);
        var boundaryParticles = m_BoundaryQuery.ToComponentDataArrayAsync<BoundaryParticleComponent>(Allocator.TempJob, out var boundaryParticleHandle);

        Dependency = JobHandle.CombineDependencies(
            Dependency,
            boundaryPositionHandle,
            boundaryParticleHandle
        );
    }

    protected override void OnDestroy()
    {

    }
}
