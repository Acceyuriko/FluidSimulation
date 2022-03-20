using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Extensions;
using Unity.Physics.Systems;
using Unity.Transforms;
using UnityEngine;

[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateBefore(typeof(BuildPhysicsWorld))]
public class FluidSPHSystem : SystemBase
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
        [ReadOnly] public float kernelRadiusRate;
        [ReadOnly] public NativeMultiHashMap<uint, int> grid;
        [ReadOnly] public NativeArray<float> gridSize;
        [ReadOnly] public NativeArray<Translation> boundaryPositions;
        [ReadOnly] public NativeArray<BoundaryParticleComponent> boundaryParticles;
        [ReadOnly] public NativeMultiHashMap<uint, int> boundaryGrid;

        public NativeArray<float> densities;
        public NativeArray<float> pressures;

        public void Execute(int index)
        {
            float density = 0f;
            float3 position;
            float radius, mass, gasConstant, restDensity;

            if (index >= positions.Length)
            {
                var particle = boundaryParticles[index - positions.Length];
                position = boundaryPositions[index - positions.Length].Value;
                radius = particle.radius;
                mass = particle.mass;
                gasConstant = particle.gasConstant;
                restDensity = particle.restDensity;
            }
            else
            {
                var particle = particles[index];
                position = positions[index].Value;
                radius = particle.radius;
                mass = 1f / masses[index].InverseMass;
                gasConstant = particle.gasConstant;
                restDensity = particle.restDensity;
            }

            var gridPosition = Quantize(position, gridSize[0]);
            float kernelRadius = radius * kernelRadiusRate;
            var kernelRadius2 = math.pow(kernelRadius, 2f);
            var poly6Constant = mass * 315f / (64f * math.PI * math.pow(kernelRadius, 9f));

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        var neighborGridIndex = Hash(gridPosition + new int3(x, y, z));
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
            // 理想气体方程 p_i = k(\rho_i - \rho_0)
            pressures[index] = math.max(gasConstant * (density - restDensity), 0);
        }

        private static float ComputeDensity(float poly6Constant, float kernelRadius2, float distance2)
        {
            if (distance2 < kernelRadius2)
            {
                // 光滑核函数 poly6, \rho_i = m \Sigma{W_{poly6}}
                // 其中 W_poly6 = \frac{315}{64 \pi h^9} (h^2 - r^2)^3
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
        [ReadOnly] public float kernelRadiusRate;
        [ReadOnly] public NativeMultiHashMap<uint, int> grid;
        [ReadOnly] public NativeArray<float> gridSize;
        [ReadOnly] public NativeArray<Translation> boundaryPositions;
        [ReadOnly] public NativeArray<BoundaryParticleComponent> boundaryParticles;
        [ReadOnly] public NativeMultiHashMap<uint, int> boundaryGrid;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> pressures;
        public NativeArray<float3> forces;


        public void Execute(int index)
        {
            float densityi = densities[index];
            float pressurei = pressures[index];

            var position = positions[index].Value;
            var gridPosition = Quantize(position, gridSize[0]);
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
                        var neighborGridIndex = Hash(gridPosition + new int3(x, y, z));
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
                                densities[positions.Length + j],
                                pressurei,
                                pressures[positions.Length + j],
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
                // 其中 \Delta{W_{spiky}} = - \frac{45}{\pi h^6}(h - r)^2e_r
                // e_r 表示 i - j 的单位向量
                float3 forcePressure = -densityi * massj * (pressurei / math.pow(densityi, 2f) + pressurej / math.pow(densityj, 2f)) *
                    (-45f / (math.PI * math.pow(kernelRadius, 6f)) * math.pow(kernelRadius - distance, 2f)) *
                    math.normalize(vij);

                // f_i^{visco} = \frac{\mu}{\rho_i}\Sigma{m_j(u_j - u_i)}\Delta^2W_{visco}
                // 其中 \Delta^2W_{visco}\Delta^2W_{visco} = \frac{45}{\pi h^6}(h - r)
                float3 forceViscosity = viscosity / densityi * massj * (velocityj - velocityi) *
                    45f / (math.PI * math.pow(kernelRadius, 6f)) * (kernelRadius - distance);

                return new float3x2 { c0 = forcePressure, c1 = forceViscosity };
            }
            return float3x2.zero;
        }
    }

    private EntityQuery m_ParticleQuery;
    private EntityQuery m_BoundaryQuery;

    private readonly int m_Concurrency = 64;

    // 核函数半径一般取粒子半径的 3 ~ 5 倍
    private readonly float m_KernelRadiusRate = 4f;

    protected override void OnCreate()
    {
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

        RequireForUpdate(m_ParticleQuery);
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

        var gridSize = new NativeArray<float>(1, Allocator.TempJob);
        gridSize[0] = float.MaxValue;

        var findGridSizeJob = new FindGridSizeJob
        {
            particleTypeHandle = GetComponentTypeHandle<FluidParticleComponent>(),
            gridSize = gridSize,
            kernelRadiusRate = m_KernelRadiusRate,
        };
        Dependency = findGridSizeJob.Schedule(m_ParticleQuery, Dependency);

        // 计算网格布局
        var min = new NativeArray<float>(3, Allocator.TempJob);
        min[0] = min[1] = min[2] = float.MaxValue;
        var max = new NativeArray<float>(3, Allocator.TempJob);
        max[0] = max[1] = max[2] = float.MinValue;

        var grid = new NativeMultiHashMap<uint, int>(positions.Length, Allocator.TempJob);
        var boundaryGrid = new NativeMultiHashMap<uint, int>(positions.Length, Allocator.TempJob);

        Dependency = Job
            .WithReadOnly(positions)
            .WithReadOnly(boundaryPositions)
            .WithReadOnly(gridSize)
            .WithCode(() =>
            {
                for (int i = 0; i < positions.Length; i++)
                {
                    if (positions[i].Value.x < min[0]) min[0] = positions[i].Value.x;
                    if (positions[i].Value.y < min[1]) min[1] = positions[i].Value.y;
                    if (positions[i].Value.z < min[2]) min[2] = positions[i].Value.z;

                    if (positions[i].Value.x > max[0]) max[0] = positions[i].Value.x;
                    if (positions[i].Value.y > max[1]) max[1] = positions[i].Value.y;
                    if (positions[i].Value.z > max[2]) max[2] = positions[i].Value.z;

                    grid.Add(Hash(Quantize(positions[i].Value, gridSize[0])), i);
                }

                for (int i = 0; i < boundaryPositions.Length; i++)
                {
                    if (boundaryPositions[i].Value.x < min[0]) min[0] = boundaryPositions[i].Value.x;
                    if (boundaryPositions[i].Value.y < min[1]) min[1] = boundaryPositions[i].Value.y;
                    if (boundaryPositions[i].Value.z < min[2]) min[2] = boundaryPositions[i].Value.z;

                    if (boundaryPositions[i].Value.x > max[0]) max[0] = boundaryPositions[i].Value.x;
                    if (boundaryPositions[i].Value.y > max[1]) max[1] = boundaryPositions[i].Value.y;
                    if (boundaryPositions[i].Value.z > max[2]) max[2] = boundaryPositions[i].Value.z;

                    boundaryGrid.Add(Hash(Quantize(boundaryPositions[i].Value, gridSize[0])), i);
                }

                min[0] = math.floor(min[0] / gridSize[0]) * gridSize[0];
                min[1] = math.floor(min[1] / gridSize[0]) * gridSize[0];
                min[2] = math.floor(min[2] / gridSize[0]) * gridSize[0];
                // 按左闭右开划分网格，因此最大值需要加上额外一个网格
                max[0] = math.ceil(max[0] / gridSize[0]) * gridSize[0] + 1f;
                max[1] = math.ceil(max[1] / gridSize[0]) * gridSize[0] + 1f;
                max[2] = math.ceil(max[2] / gridSize[0]) * gridSize[0] + 1f;
            })
            .Schedule(Dependency);

        var pressures = new NativeArray<float>(positions.Length + boundaryPositions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var densities = new NativeArray<float>(positions.Length + boundaryPositions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        // 计算密度和压力
        var computeDensityAndPressureJob = new ComputeDensityAndPressureJob
        {
            positions = positions,
            masses = physicsMasses,
            particles = particles,
            kernelRadiusRate = m_KernelRadiusRate,
            grid = grid,
            gridSize = gridSize,
            boundaryPositions = boundaryPositions,
            boundaryParticles = boundaryParticles,
            boundaryGrid = boundaryGrid,

            densities = densities,
            pressures = pressures,
        };
        Dependency = computeDensityAndPressureJob.Schedule(positions.Length + boundaryPositions.Length, m_Concurrency, Dependency);

        var forces = new NativeArray<float3>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        // 计算力
        var computeForceJob = new ComputeForceJob
        {
            positions = positions,
            masses = physicsMasses,
            velocities = physicsVelocities,
            particles = particles,
            kernelRadiusRate = m_KernelRadiusRate,
            grid = grid,
            gridSize = gridSize,
            boundaryPositions = boundaryPositions,
            boundaryParticles = boundaryParticles,
            boundaryGrid = boundaryGrid,
            densities = densities,
            pressures = pressures,

            forces = forces,
        };
        Dependency = computeForceJob.Schedule(positions.Length, m_Concurrency, Dependency);

        var deltaTime = Time.DeltaTime;

        Dependency = Entities
            .WithAll<FluidParticleComponent>()
            .ForEach((int entityInQueryIndex, ref PhysicsVelocity velocity, in PhysicsMass mass) =>
            {
                // SPH 中 F_sph = \rho a ， 量纲与一般的 F 不同，因此需要转换。 F_unity = F_sph * mass / \rho
                velocity.ApplyLinearImpulse(mass, forces[entityInQueryIndex] / densities[entityInQueryIndex] / mass.InverseMass * deltaTime);
            })
            .ScheduleParallel(Dependency);

        gridSize.Dispose(Dependency);
        positions.Dispose(Dependency);
        particles.Dispose(Dependency);
        physicsMasses.Dispose(Dependency);
        physicsVelocities.Dispose(Dependency);
        min.Dispose(Dependency);
        max.Dispose(Dependency);
        grid.Dispose(Dependency);
        boundaryPositions.Dispose(Dependency);
        boundaryParticles.Dispose(Dependency);
        boundaryGrid.Dispose(Dependency);

        pressures.Dispose(Dependency);
        densities.Dispose(Dependency);
        forces.Dispose(Dependency);
    }

    protected override void OnDestroy()
    {
    }

    private static int3 Quantize(float3 position, float size)
    {
        return new int3(
            (int)math.floor(position.x / size),
            (int)math.floor(position.y / size),
            (int)math.floor(position.z / size)
        );
    }

    // FNV-1 hash https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function 
    private static uint Hash(int3 p)
    {
        uint hash = 2166136261u;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < sizeof(int); j++)
            {
                byte b = (byte)(p[i] >> (j * 8));
                hash *= 16777619u;
                hash ^= b;
            }
        }

        return hash;
    }
}
