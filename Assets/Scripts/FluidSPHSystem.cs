using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Transforms;

[UpdateInGroup(typeof(InitializationSystemGroup))]
[UpdateBefore(typeof(EndInitializationEntityCommandBufferSystem))]
public class FluidSPHSystem : SystemBase
{
    [BurstCompile]
    private struct HashPositionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public int3 gridLength;
        [ReadOnly] public float size;
        [ReadOnly] public float3 min;
        public NativeMultiHashMap<int, int>.ParallelWriter hashMap;
        public NativeArray<int> positionToGrid;

        public void Execute(int index)
        {
            var gridIndex = HashPositionToGridIndex(positions[index].Value, min, size, gridLength);
            positionToGrid[index] = gridIndex;
            hashMap.Add(gridIndex, index);
        }
    }

    [BurstCompile]
    private struct ComputeDensityAndPressureJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public NativeArray<int> positionToGrid;
        [ReadOnly] public NativeArray<FluidParticleComponent> particles;
        [ReadOnly] public int3 gridLength;
        [ReadOnly] public NativeMultiHashMap<int, int> hashMap;
        public NativeArray<float> densities;
        public NativeArray<float> pressures;

        public void Execute(int index)
        {
            float density = 0f;
            var gridIndex = positionToGrid[index];
            var position = positions[index].Value;
            var setting = particles[index];
            var radiusSq = math.pow(setting.radius, 2);

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= -1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        var neighborGridIndex = gridIndex + XyzToGridIndex(x, y, z, gridLength);
                        var found = hashMap.TryGetFirstValue(neighborGridIndex, out var j, out var iterator);
                        while (found)
                        {
                            float distanceSq = math.lengthsq(position - positions[j].Value);
                            if (distanceSq < radiusSq)
                            {
                            }
                            found = hashMap.TryGetNextValue(out j, ref iterator);
                        }
                    }
                }
            }

            densities[index] = density;
            pressures[index] = density;
        }
    }

    [BurstCompile]
    private struct ComputeForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public int3 gridLength;
        [ReadOnly] public float size;
        [ReadOnly] public float3 min;
        [ReadOnly] public NativeMultiHashMap<int, int> hashMap;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float> pressures;

        public void Execute(int index)
        {
        }
    }


    private EndInitializationEntityCommandBufferSystem m_EndInitializationECB;

    private EntityQuery m_ParticleQuery;

    private readonly int m_Concurrency = 32;

    protected override void OnCreate()
    {
        m_EndInitializationECB = World.GetOrCreateSystem<EndInitializationEntityCommandBufferSystem>();
        m_ParticleQuery = GetEntityQuery(new EntityQueryDesc
        {
            All = new ComponentType[] { typeof(FluidParticleComponent), typeof(Translation), typeof(PhysicsVelocity), typeof(PhysicsMass) }
        });

        RequireForUpdate(m_ParticleQuery);
    }

    protected override void OnUpdate()
    {
        var commandBuffer = m_EndInitializationECB.CreateCommandBuffer().AsParallelWriter();

        var positions = m_ParticleQuery.ToComponentDataArrayAsync<Translation>(Allocator.TempJob, out var positionHandle);
        var particles = m_ParticleQuery.ToComponentDataArrayAsync<FluidParticleComponent>(Allocator.TempJob, out var particleHandle);
        var velocities = m_ParticleQuery.ToComponentDataArrayAsync<PhysicsVelocity>(Allocator.TempJob, out var velocityHandle);
        var masses = m_ParticleQuery.ToComponentDataArrayAsync<PhysicsMass>(Allocator.TempJob, out var massHandle);

        Dependency = JobHandle.CombineDependencies(
            JobHandle.CombineDependencies(
                positionHandle,
                particleHandle,
                velocityHandle
            ),
            massHandle
        );

        var gridSize = 1f;

        Entities
            .WithAll<FluidComponent>()
            .ForEach((in FluidComponent fluid) =>
            {
                var size = fluid.radius * 4f;
                if (size < gridSize)
                {
                    gridSize = size;
                }
            }).Run();

        // 计算网格布局
        var min = new float3(float.MaxValue);
        var max = new float3(float.MinValue);
        var gridLength = new int3();

        Dependency = Job.WithCode(() =>
        {
            for (int i = 0; i < positions.Length; i++)
            {
                if (positions[i].Value.x < min.x) min.x = positions[i].Value.x;
                if (positions[i].Value.y < min.y) min.y = positions[i].Value.y;
                if (positions[i].Value.z < min.z) min.z = positions[i].Value.z;

                if (positions[i].Value.x > max.x) max.x = positions[i].Value.x;
                if (positions[i].Value.y > max.y) max.y = positions[i].Value.y;
                if (positions[i].Value.z > max.z) max.z = positions[i].Value.z;
            }

            min.x = math.floor(min.x / gridSize) * gridSize;
            min.y = math.floor(min.y / gridSize) * gridSize;
            min.z = math.floor(min.z / gridSize) * gridSize;
            // 按左闭右开划分网格，因此最大值需要加上额外一个网格
            max.x = math.ceil(max.x / gridSize) * gridSize + 1f;
            max.y = math.ceil(max.y / gridSize) * gridSize + 1f;
            max.z = math.ceil(max.z / gridSize) * gridSize + 1f;

            gridLength.x = (int)math.ceil((max.x - min.x) / gridSize);
            gridLength.y = (int)math.ceil((max.y - min.y) / gridSize);
            gridLength.z = (int)math.ceil((max.z - min.z) / gridSize);
        }).Schedule(Dependency);

        // 创建网格 TODO: 使用 https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function 合并计算与创建网格
        var grid = new NativeMultiHashMap<int, int>(positions.Length, Allocator.TempJob);
        var positionToGrid = new NativeArray<int>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var hashPositionJob = new HashPositionJob
        {
            positions = positions,
            positionToGrid = positionToGrid,
            min = min,
            size = gridSize,
            gridLength = gridLength,
            hashMap = grid.AsParallelWriter(),
        };
        Dependency = hashPositionJob.Schedule(positions.Length, m_Concurrency, Dependency);

        var pressures = new NativeArray<float>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var densities = new NativeArray<float>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        // 计算密度和压力
        var computeDensityAndPressureJob = new ComputeDensityAndPressureJob
        {
            positions = positions,
            positionToGrid = positionToGrid,
            particles = particles,
            gridLength = gridLength,
            hashMap = grid,
            densities = densities,
            pressures = pressures,
        };
        Dependency = computeDensityAndPressureJob.Schedule(positions.Length, m_Concurrency, Dependency);

        var forces = new NativeArray<float3>(positions.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);


        // 计算力并应用
        var computeForceJob = new ComputeForceJob
        {
            positions = positions,
            min = min,
            size = gridSize,
            gridLength = gridLength,
            hashMap = grid,
            densities = densities,
            pressures = pressures,
        };
        Dependency = computeForceJob.Schedule(positions.Length, m_Concurrency, Dependency);

        positions.Dispose(Dependency);
        grid.Dispose(Dependency);
        positionToGrid.Dispose(Dependency);

        pressures.Dispose(Dependency);
        densities.Dispose(Dependency);
        particles.Dispose(Dependency);

        forces.Dispose(Dependency);
        velocities.Dispose(Dependency);
        masses.Dispose(Dependency);

        m_EndInitializationECB.AddJobHandleForProducer(Dependency);
    }

    protected override void OnDestroy()
    {
    }

    private static int HashPositionToGridIndex(float3 position, float3 min, float size, int3 gridLength)
    {
        return XyzToGridIndex(
            (int)math.floor((position.x - min.x) / size),
            (int)math.floor((position.y - min.y) / size),
            (int)math.floor((position.z - min.z) / size),
            gridLength
        );
    }

    private static int XyzToGridIndex(int x, int y, int z, int3 gridLength)
    {
        return x + gridLength.x * y + gridLength.x * gridLength.y * z;
    }
}
