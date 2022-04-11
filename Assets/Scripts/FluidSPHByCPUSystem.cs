using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Systems;
using Unity.Transforms;
using UnityEngine;

[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateBefore(typeof(BuildPhysicsWorld))]
public partial class FluidSPHByCPUSystem : SystemBase
{
    [BurstCompile]
    private struct FindGridSizeJob : IJobEntityBatch
    {
        public ComponentTypeHandle<FluidParticleComponent> particleTypeHandle;
        public NativeArray<float> gridSize;
        public float kernelRadiusRate;

        public void Execute(ArchetypeChunk batchInChunk, int batchIndex)
        {
            var settings = batchInChunk.GetNativeArray(particleTypeHandle);
            var radius = settings[0].radius;
            if (radius < gridSize[0] / kernelRadiusRate)
            {
                gridSize[0] = radius * kernelRadiusRate;
            }
        }
    }

    [BurstCompile]
    private struct ComputeDensityAndPressureJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public NativeArray<PhysicsMass> masses;
        [ReadOnly] public NativeArray<FluidParticleComponent> particles;
        [ReadOnly] public NativeMultiHashMap<uint, int> grid;
        [ReadOnly] public NativeArray<float> gridSize;
        [ReadOnly] public NativeArray<Translation> boundaryPositions;
        [ReadOnly] public NativeMultiHashMap<uint, int> boundaryGrid;
        [ReadOnly] public float kernelRadiusRate;

        public NativeArray<float> densities;
        public NativeArray<float> pressures;

        public void Execute(int index)
        {
            float density = 0f;
            var particle = particles[index];
            float3 position = positions[index].Value;
            float radius = particle.radius;
            float mass = 1f / masses[index].InverseMass;
            float gasConstant = particle.gasConstant;
            float restDensity = particle.restDensity;
            var gridPosition = FluidSPHUtils.Quantize(position, gridSize[0]);
            float kernelRadius = radius * kernelRadiusRate;
            var kernelRadius2 = math.pow(kernelRadius, 2f);
            var poly6Constant = mass * 315f / (64f * math.PI * math.pow(kernelRadius, 9f));

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        var neighborGridIndex = FluidSPHUtils.Hash(gridPosition + new int3(x, y, z));
                        var found = grid.TryGetFirstValue(neighborGridIndex, out var j, out var iterator);
                        while (found)
                        {
                            float distance2 = math.lengthsq(position - positions[j].Value);
                            density += ComputeDensity(poly6Constant, kernelRadius2, distance2);

                            found = grid.TryGetNextValue(out j, ref iterator);
                        }

                        found = boundaryGrid.TryGetFirstValue(neighborGridIndex, out j, out iterator);
                        while (found)
                        {
                            float distance2 = math.lengthsq(position - boundaryPositions[j].Value);
                            density += ComputeDensity(poly6Constant, kernelRadius2, distance2);

                            found = boundaryGrid.TryGetNextValue(out j, ref iterator);
                        }
                    }
                }
            }

            densities[index] = density;
            // ideal gas state eq. p_i = k(\rho_i - \rho_0)
            pressures[index] = math.max(gasConstant * (density - restDensity), 0);
        }

        private static float ComputeDensity(float poly6Constant, float kernelRadius2, float distance2)
        {
            if (distance2 < kernelRadius2)
            {
                // kernel poly6, \rho_i = m \Sigma{W_{poly6}}
                // where W_poly6 = \frac{315}{64 \pi h^9} (h^2 - r^2)^3
                return poly6Constant * math.pow(kernelRadius2 - distance2, 3f);
            }
            return 0;
        }
    }


    [BurstCompile]
    private struct ComputeForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public NativeArray<PhysicsMass> masses;
        [ReadOnly] public NativeArray<PhysicsVelocity> velocities;
        [ReadOnly] public NativeArray<FluidParticleComponent> particles;
        [ReadOnly] public NativeMultiHashMap<uint, int> grid;
        [ReadOnly] public NativeArray<float> gridSize;
        [ReadOnly] public NativeArray<Translation> boundaryPositions;
        [ReadOnly] public NativeArray<BoundaryParticleComponent> boundaryParticles;
        [ReadOnly] public NativeMultiHashMap<uint, int> boundaryGrid;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> pressures;
        [ReadOnly] public float kernelRadiusRate;

        public NativeArray<float3> forces;

        public void Execute(int index)
        {
            float densityi = densities[index];
            float pressurei = pressures[index];

            var position = positions[index].Value;
            var gridPosition = FluidSPHUtils.Quantize(position, gridSize[0]);
            float kernelRadius = particles[index].radius * kernelRadiusRate;

            float viscosity = particles[index].viscosity;
            float3 velocityi = velocities[index].Linear;

            float3 forcePressure = new float3();
            float3 forceViscosity = new float3();
            float3 forceGravity = new float3(0, -9.81f, 0) * densityi;

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        var neighborGridIndex = FluidSPHUtils.Hash(gridPosition + new int3(x, y, z));
                        var found = grid.TryGetFirstValue(neighborGridIndex, out var j, out var iterator);
                        while (found)
                        {
                            if (index != j)
                            {
                                float3 vij = position - positions[j].Value;
                                var result = ComputeForcePressureAndViscosity(
                                    math.length(vij),
                                    kernelRadius,
                                    viscosity,
                                    densityi,
                                    densities[j],
                                    pressurei,
                                    pressures[j],
                                    velocityi,
                                    velocities[j].Linear,
                                    1f / masses[j].InverseMass,
                                    vij
                                );
                                forcePressure += result.c0;
                                forceViscosity += result.c1;
                            }
                            found = grid.TryGetNextValue(out j, ref iterator);
                        }

                        found = boundaryGrid.TryGetFirstValue(neighborGridIndex, out j, out iterator);
                        while (found)
                        {
                            float3 vij = position - boundaryPositions[j].Value;

                            var result = ComputeForcePressureAndViscosity(
                                math.length(vij),
                                kernelRadius,
                                viscosity,
                                densityi,
                                boundaryParticles[j].restDensity,
                                pressurei,
                                0,
                                velocityi,
                                new float3(0),
                                boundaryParticles[j].mass,
                                vij
                            );
                            forcePressure += result.c0;
                            forceViscosity += result.c1;

                            found = boundaryGrid.TryGetNextValue(out j, ref iterator);
                        }

                    }
                }
            }

            forces[index] = forcePressure + forceViscosity + forceGravity;
        }

        private static float3x2 ComputeForcePressureAndViscosity(
            float distance,
            float kernelRadius,
            float viscosity,
            float densityi,
            float densityj,
            float pressurei,
            float pressurej,
            float3 velocityi,
            float3 velocityj,
            float massj,
            float3 vij
        )
        {
            if (distance < kernelRadius)
            {
                // f_{i}^{press} = - \rho_i \Sigma{ m_j (\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2})\Delta{W_{spiky}}} 
                // where \Delta{W_{spiky}} = - \frac{45}{\pi h^6}(h - r)^2e_r
                // e_r is i - j 
                float3 forcePressure = -densityi * massj * (pressurei / math.pow(densityi, 2f) + pressurej / math.pow(densityj, 2f)) *
                    (-45f / (math.PI * math.pow(kernelRadius, 6f)) * math.pow(kernelRadius - distance, 2f)) *
                    math.normalize(vij);

                // f_i^{visco} = \frac{\mu}{\rho_i}\Sigma{m_j(u_j - u_i)}\Delta^2W_{visco}
                // where \Delta^2W_{visco}\Delta^2W_{visco} = \frac{45}{\pi h^6}(h - r)
                float3 forceViscosity = viscosity / densityi * massj * (velocityj - velocityi) *
                    45f / (math.PI * math.pow(kernelRadius, 6f)) * (kernelRadius - distance);

                return new float3x2 { c0 = forcePressure, c1 = forceViscosity };
            }
            return float3x2.zero;
        }
    }

    private EntityQuery m_ParticleQuery;
    private EntityQuery m_BoundaryQuery;

    // can not be more, maybe bug of unity.entities
    private readonly int m_Concurrency = 12;

    protected override void OnCreate()
    {
        m_ParticleQuery = GetEntityQuery(FluidSPHUtils.ParticleQueryComponentTypes);
        m_BoundaryQuery = GetEntityQuery(FluidSPHUtils.BoundaryQueryComponentTypes);

        RequireForUpdate(m_ParticleQuery);
        RequireSingletonForUpdate<SimulationSettings>();
    }

    protected override void OnDestroy()
    {
    }

    protected override void OnUpdate()
    {
        var settings = GetSingleton<SimulationSettings>();
        if (SystemInfo.supportsComputeShaders && settings.UseGPU)
        {
            Enabled = false;
            return;
        }
        World.GetOrCreateSystem<FixedStepSimulationSystemGroup>().Timestep = 1f / settings.FPS;

        var data = FluidSPHUtils.InitializeData(m_ParticleQuery, m_BoundaryQuery, GetComponentTypeHandle<FluidParticleComponent>(true));
        var positions = data.positions;
        var particles = data.particles;
        var physicsMasses = data.physicsMasses;
        var physicsVelocities = data.physicsVelocities;
        var boundaryPositions = data.boundaryPositions;
        var boundaryParticles = data.boundaryParticles;
        var gridSize = data.gridSize;
        var grid = data.grid;
        var boundaryGrid = data.boundaryGrid;

        var pressures = new NativeArray<float>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var densities = new NativeArray<float>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        var computeDensityAndPressureJob = new ComputeDensityAndPressureJob
        {
            positions = positions,
            masses = physicsMasses,
            particles = particles,
            grid = grid,
            gridSize = gridSize,
            boundaryPositions = boundaryPositions,
            boundaryGrid = boundaryGrid,
            kernelRadiusRate = FluidSPHUtils.KernelRadiusRate,

            densities = densities,
            pressures = pressures,
        };
        Dependency = computeDensityAndPressureJob.Schedule(positions.Length, m_Concurrency, data.dependency);

        var forces = new NativeArray<float3>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        var computeForceJob = new ComputeForceJob
        {
            positions = positions,
            masses = physicsMasses,
            velocities = physicsVelocities,
            particles = particles,
            grid = grid,
            gridSize = gridSize,
            boundaryPositions = boundaryPositions,
            boundaryParticles = boundaryParticles,
            boundaryGrid = boundaryGrid,
            densities = densities,
            pressures = pressures,
            kernelRadiusRate = FluidSPHUtils.KernelRadiusRate,

            forces = forces,
        };
        Dependency = computeForceJob.Schedule(positions.Length, m_Concurrency, Dependency);

        var applyForceJob = new FluidSPHUtils.ApplyForceJob
        {
            densities = densities,
            forces = forces,
            deltaTime = Time.DeltaTime,
            massTypeHandle = GetComponentTypeHandle<PhysicsMass>(true),
            velocityTypeHandle = GetComponentTypeHandle<PhysicsVelocity>(),
        };
        Dependency = applyForceJob.ScheduleParallel(m_ParticleQuery, Dependency);

        FluidSPHUtils.Dispose(data, Dependency);
        pressures.Dispose(Dependency);
        densities.Dispose(Dependency);
        forces.Dispose(Dependency);
    }
}
