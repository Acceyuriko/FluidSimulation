using System;
using System.Linq;
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
        [ReadOnly] public NativeMultiHashMap<uint, int> hashMap;
        [ReadOnly] public NativeArray<float> gridSize;
        public NativeArray<float> densities;
        public NativeArray<float> pressures;

        public void Execute(int index)
        {
            float density = 0f;
            var position = positions[index].Value;
            var setting = particles[index];
            var gridPosition = Quantize(positions[index].Value, gridSize[0]);
            float kernelRadius = setting.radius * kernelRadiusRate;
            var kernelRadius2 = math.pow(kernelRadius, 2f);
            var poly6Constant = 1f / masses[index].InverseMass * 315f / (64f * math.PI * math.pow(kernelRadius, 9f));

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        var neighborGridIndex = Hash(gridPosition + new int3(x, y, z));
                        var found = hashMap.TryGetFirstValue(neighborGridIndex, out var j, out var iterator);
                        while (found)
                        {
                            float distance2 = math.lengthsq(position - positions[j].Value);
                            if (distance2 < kernelRadius2)
                            {
                                // 光滑核函数 poly6, \rho_i = m \Sigma{W_{poly6}}
                                // 其中 W_poly6 = \frac{315}{64 \pi h^9} (h^2 - r^2)^3
                                density += poly6Constant * math.pow(kernelRadius2 - distance2, 3f);
                            }
                            found = hashMap.TryGetNextValue(out j, ref iterator);
                        }
                    }
                }
            }

            densities[index] = density;
            // 理想气体方程 p_i = k(\rho_i - \rho_0)
            pressures[index] = math.max(setting.gasConstant * (density - setting.density), 0);
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
        [ReadOnly] public NativeMultiHashMap<uint, int> hashMap;
        [ReadOnly] public NativeArray<float> gridSize;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> pressures;
        public NativeArray<float3> forces;


        public void Execute(int index)
        {
            var position = positions[index].Value;
            var gridPosition = Quantize(position, gridSize[0]);
            float kernelRadius = particles[index].radius * kernelRadiusRate;

            float density = densities[index];
            float pressure = pressures[index];
            float viscosity = particles[index].viscosity;
            float3 velocity = velocities[index].Linear;

            float3 forcePressure = new float3();
            float3 forceViscosity = new float3();
            float3 forceGravity = new float3(0, -9.81f, 0) * density;

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        var neighborGridIndex = Hash(gridPosition + new int3(x, y, z));
                        var found = hashMap.TryGetFirstValue(neighborGridIndex, out var j, out var iterator);
                        while (found)
                        {
                            if (index != j)
                            {
                                float3 vij = position - positions[j].Value;
                                float distance = math.length(vij);
                                float densityj = densities[j];
                                float pressurej = pressures[j];
                                float massj = 1f / masses[j].InverseMass;
                                float3 velocityj = velocities[j].Linear;

                                if (distance < kernelRadius)
                                {
                                    // f_{i}^{press} = - \rho_i \Sigma{ m_j (\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2})\Delta{W_{spiky}}} 
                                    // 其中 \Delta{W_{spiky}} = - \frac{45}{\pi h^6}(h - r)^2e_r
                                    // e_r 表示 i - j 的单位向量
                                    forcePressure += -density * massj * (pressure / math.pow(density, 2f) + pressurej / math.pow(densityj, 2f)) *
                                        (-45f / (math.PI * math.pow(kernelRadius, 6f)) * math.pow(kernelRadius - distance, 2f)) *
                                        math.normalize(vij);

                                    // f_i^{visco} = \frac{\mu}{\rho_i}\Sigma{m_j(u_j - u_i)}\Delta^2W_{visco}
                                    // 其中 \Delta^2W_{visco}\Delta^2W_{visco} = \frac{45}{\pi h^6}(h - r)
                                    forceViscosity += viscosity / density * massj * (velocityj - velocity) *
                                        45f / (math.PI * math.pow(kernelRadius, 6f)) * (kernelRadius - distance);
                                }
                            }
                            found = hashMap.TryGetNextValue(out j, ref iterator);
                        }
                    }
                }
            }

            forces[index] = forcePressure + forceViscosity + forceGravity;
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

        Dependency = Job.WithCode(() =>
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

            min[0] = math.floor(min[0] / gridSize[0]) * gridSize[0];
            min[1] = math.floor(min[1] / gridSize[0]) * gridSize[0];
            min[2] = math.floor(min[2] / gridSize[0]) * gridSize[0];
            // 按左闭右开划分网格，因此最大值需要加上额外一个网格
            max[0] = math.ceil(max[0] / gridSize[0]) * gridSize[0] + 1f;
            max[1] = math.ceil(max[1] / gridSize[0]) * gridSize[0] + 1f;
            max[2] = math.ceil(max[2] / gridSize[0]) * gridSize[0] + 1f;
        }).Schedule(Dependency);

        var pressures = new NativeArray<float>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var densities = new NativeArray<float>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        // 计算密度和压力
        var computeDensityAndPressureJob = new ComputeDensityAndPressureJob
        {
            positions = positions,
            masses = physicsMasses,
            particles = particles,
            kernelRadiusRate = m_KernelRadiusRate,
            hashMap = grid,
            gridSize = gridSize,
            densities = densities,
            pressures = pressures,
        };
        Dependency = computeDensityAndPressureJob.Schedule(positions.Length, m_Concurrency, Dependency);

        var forces = new NativeArray<float3>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        // 计算力
        var computeForceJob = new ComputeForceJob
        {
            positions = positions,
            masses = physicsMasses,
            velocities = physicsVelocities,
            particles = particles,
            kernelRadiusRate = m_KernelRadiusRate,
            hashMap = grid,
            gridSize = gridSize,
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
