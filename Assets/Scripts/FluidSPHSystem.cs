using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Extensions;
using Unity.Transforms;
using UnityEngine;

[UpdateInGroup(typeof(InitializationSystemGroup))]
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

        public void Execute(int index)
        {
            hashMap.Add(HashPositionToGridIndex(positions[index].Value, min, size, gridLength), index);
        }
    }

    [BurstCompile]
    private struct ComputeDensityAndPressureJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public int3 gridLength;
        [ReadOnly] public float size;
        [ReadOnly] public float3 min;
        [ReadOnly] public NativeMultiHashMap<int, int> hashMap;
        public NativeArray<float> densities;
        public NativeArray<float> pressures;

        public void Execute(int index)
        {

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

    private int m_Concurrency;

    private readonly List<FluidParticleComponent> m_FluidTypes = new List<FluidParticleComponent>();

    private readonly float m_GridSize = 1f;

    protected override void OnCreate()
    {
        m_Concurrency = Environment.ProcessorCount;
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
        EntityManager.GetAllUniqueSharedComponentData(m_FluidTypes);

        var positions = m_ParticleQuery.ToComponentDataArray<Translation>(Allocator.TempJob);

        // 计算网格布局
        var bounds = new Bounds();
        bounds.SetMinMax(new Vector3(float.MinValue, float.MinValue, float.MinValue), new Vector3(float.MaxValue, float.MaxValue, float.MaxValue));

        var gridSize = m_GridSize;
        var gridLength = new int3();

        var min = bounds.min;
        var max = bounds.max;
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

        bounds.SetMinMax(min, max);

        gridLength.x = (int)math.ceil((max.x - min.x) / gridSize);
        gridLength.y = (int)math.ceil((max.y - min.y) / gridSize);
        gridLength.z = (int)math.ceil((max.z - min.z) / gridSize);

        // 创建网格
        var grid = new NativeMultiHashMap<int, int>(positions.Length, Allocator.TempJob);
        var hashPositionJob = new HashPositionJob
        {
            positions = positions,
            min = bounds.min,
            size = gridSize,
            gridLength = gridLength,
            hashMap = grid.AsParallelWriter(),
        };
        Dependency = hashPositionJob.Schedule(positions.Length, m_Concurrency, Dependency);

        var pressures = new NativeArray<float>(positions.Length, Allocator.TempJob);
        var densities = new NativeArray<float>(positions.Length, Allocator.TempJob);

        // 计算密度和压力
        var computeDensityAndPressureJob = new ComputeDensityAndPressureJob
        {
            positions = positions,
            min = bounds.min,
            size = gridSize,
            gridLength = gridLength,
            hashMap = grid,
            densities = densities,
            pressures = pressures,
        };
        Dependency = computeDensityAndPressureJob.Schedule(positions.Length, m_Concurrency, Dependency);

        var forces = new NativeArray<float3>(positions.Length, Allocator.TempJob);

        var velocities = m_ParticleQuery.ToComponentDataArray<PhysicsVelocity>(Allocator.TempJob);
        var masses = m_ParticleQuery.ToComponentDataArray<PhysicsMass>(Allocator.TempJob);

        // var velocity = velocities[0];
        // Debug.Log(velocity.Linear);
        // velocity.ApplyLinearImpulse(masses[0], new float3(0, 1, 0));
        // velocities[0] = velocity;

        // 计算力并应用
        var computeForceJob = new ComputeForceJob
        {
            positions = positions,
            min = bounds.min,
            size = gridSize,
            gridLength = gridLength,
            hashMap = grid,
            densities = densities,
            pressures = pressures,
        };
        Dependency = computeForceJob.Schedule(positions.Length, m_Concurrency, Dependency);

        positions.Dispose(Dependency);
        grid.Dispose(Dependency);
        pressures.Dispose(Dependency);
        densities.Dispose(Dependency);
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
        return (int)math.floor((position.x - min.x) / size) +
            gridLength.x * (int)math.floor((position.y - min.y) / size) +
            gridLength.x * gridLength.y * (int)math.floor((position.z - min.z) / size);
    }
}
