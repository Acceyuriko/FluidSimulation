using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEngine;

[UpdateInGroup(typeof(SimulationSystemGroup))]
[UpdateBefore(typeof(FixedStepSimulationSystemGroup))]
public class FluidSPHSystem : SystemBase
{
    [BurstCompile]
    private struct HashPositionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public NativeArray<float3> bounds;
        [ReadOnly] public float size;

        public void Execute(int index)
        {

        }
    }

    private BeginFixedStepSimulationEntityCommandBufferSystem m_BeginFixedStepECB;

    private EntityQuery m_ParticleQuery;

    private int m_Concurrency;

    private readonly List<FluidParticleComponent> m_FluidTypes = new List<FluidParticleComponent>();

    private readonly float m_GridSize = 1f;

    protected override void OnCreate()
    {
        m_Concurrency = Environment.ProcessorCount;
        m_BeginFixedStepECB = World.GetOrCreateSystem<BeginFixedStepSimulationEntityCommandBufferSystem>();
        m_ParticleQuery = GetEntityQuery(new EntityQueryDesc
        {
            All = new ComponentType[] { typeof(FluidParticleComponent), typeof(Translation) }
        });

        RequireForUpdate(m_ParticleQuery);
    }

    protected override void OnUpdate()
    {
        var commandBuffer = m_BeginFixedStepECB.CreateCommandBuffer().AsParallelWriter();
        EntityManager.GetAllUniqueSharedComponentData(m_FluidTypes);

        var positions = m_ParticleQuery.ToComponentDataArrayAsync<Translation>(Allocator.TempJob, out JobHandle positionHandle);
        var boundsArray = new NativeArray<float3>(2, Allocator.TempJob);
        boundsArray[0] = new float3(float.MaxValue);
        boundsArray[1] = new float3(float.MinValue);

        var gridSize = m_GridSize;

        Dependency = Job.WithCode(() =>
        {
            var min = boundsArray[0];
            var max = boundsArray[1];
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

            boundsArray[0] = min;
            boundsArray[1] = max;
        }).Schedule(positionHandle);

        var gridLength = new NativeArray<int>(3, Allocator.TempJob);
        gridLength[0] = (int)((boundsArray[1].x - boundsArray[0].x) / gridSize);
        gridLength[1] = (int)((boundsArray[1].y - boundsArray[0].y) / gridSize);
        gridLength[2] = (int)((boundsArray[1].z - boundsArray[0].z) / gridSize);

        var grid = new NativeArray<NativeList<int>>(gridLength[0] * gridLength[1] * gridLength[2], Allocator.TempJob);

        var hashPositionJob = new HashPositionJob
        {
            positions = positions,
            bounds = boundsArray,
            size = gridSize,
        };
        Dependency = hashPositionJob.Schedule(positions.Length, m_Concurrency, Dependency);

        positions.Dispose(Dependency);
        boundsArray.Dispose(Dependency);
        gridLength.Dispose(Dependency);
        grid.Dispose(Dependency);

        m_BeginFixedStepECB.AddJobHandleForProducer(Dependency);
    }

    protected override void OnDestroy()
    {
    }

    private static int HashPositionToGridIndex(float3 position, float3 min, float size, NativeArray<int> gridLength)
    {
        return (int)((position.x - min.x) / size) +
            gridLength[0] * (int)((position.y - min.y) / size) +
            gridLength[0] * gridLength[1] * (int)((position.z - min.z) / size);
    }
}
