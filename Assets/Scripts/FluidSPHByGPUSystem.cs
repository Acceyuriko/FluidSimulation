using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Transforms;
using UnityEngine;
using UnityEngine.Rendering;

public partial class FluidSPHByGPUSystem : SystemBase
{
    private readonly int NUM_THREADS = 64;
    private readonly int BITONIC_FLIP = 0;
    private readonly int BITONIC_DISPERSE = 1;

    private EntityQuery m_ParticleQuery;
    private EntityQuery m_BoundaryQuery;

    private int m_ParticleCount = -1;

    private NativeArray<float3> m_Positions = new NativeArray<float3>(0, Allocator.Persistent);
    private ComputeBuffer m_PositionsBuffer;

    private NativeArray<float> m_Masses = new NativeArray<float>(0, Allocator.Persistent);
    private ComputeBuffer m_MassesBuffer;

    private NativeArray<float3> m_Velocities = new NativeArray<float3>(0, Allocator.Persistent);
    private ComputeBuffer m_VelocitiesBuffer;

    private NativeArray<FluidParticleComponent> m_Particles = new NativeArray<FluidParticleComponent>(0, Allocator.Persistent);
    private ComputeBuffer m_ParticlesBuffer;

    private int m_BoundaryCount = -1;

    private NativeArray<float3> m_BoundaryPositions = new NativeArray<float3>(0, Allocator.Persistent);
    private ComputeBuffer m_BoundaryPositionsBuffer;

    private NativeArray<BoundaryParticleComponent> m_BoundaryParticles = new NativeArray<BoundaryParticleComponent>(0, Allocator.Persistent);
    private ComputeBuffer m_BoundaryParticlesBuffer;

    private int m_IndexMapCount = 0;
    private ComputeBuffer m_IndexMap;

    private int m_TableCount = 0;
    private ComputeBuffer m_Table;

    private ComputeBuffer m_Densities;

    private ComputeBuffer m_Pressures;

    private ComputeShader m_Shader;


    protected override void OnCreate()
    {
        m_ParticleQuery = GetEntityQuery(FluidSPHUtils.ParticleQueryComponentTypes);
        m_BoundaryQuery = GetEntityQuery(FluidSPHUtils.BoundaryQueryComponentTypes);

        m_Shader = Resources.Load("FluidSolver") as ComputeShader;

        RequireForUpdate(m_ParticleQuery);
        RequireSingletonForUpdate<SimulationSettings>();
    }

    protected override void OnUpdate()
    {
        var settings = GetSingleton<SimulationSettings>();
        if (!SystemInfo.supportsComputeShaders || !settings.UseGPU)
        {
            Enabled = false;
            return;
        }
        World.GetOrCreateSystem<FixedStepSimulationSystemGroup>().Timestep = 1f / settings.FPS;

        var gridSize = new NativeArray<float>(1, Allocator.TempJob);
        gridSize[0] = float.MaxValue;

        var findGridSizeJob = new FluidSPHUtils.FindGridSizeJob
        {
            particleTypeHandle = GetComponentTypeHandle<FluidParticleComponent>(true),
            gridSize = gridSize,
            kernelRadiusRate = FluidSPHUtils.KernelRadiusRate,
        };
        var findGridSizeJobHandle = findGridSizeJob.Schedule(m_ParticleQuery, Dependency);

        var min = new NativeArray<float>(3, Allocator.TempJob);
        min[0] = min[1] = min[2] = float.MaxValue;
        var max = new NativeArray<float>(3, Allocator.TempJob);
        max[0] = max[1] = max[2] = float.MinValue;

        bool changed = false;
        if (m_ParticleCount != m_ParticleQuery.CalculateEntityCount())
        {
            m_ParticleCount = m_ParticleQuery.CalculateEntityCount();

            m_Positions.Dispose();
            m_Positions = new NativeArray<float3>(m_ParticleCount, Allocator.Persistent);
            m_PositionsBuffer?.Release();
            m_PositionsBuffer = new ComputeBuffer(m_ParticleCount, Marshal.SizeOf(typeof(float3)));

            m_Masses.Dispose();
            m_Masses = new NativeArray<float>(m_ParticleCount, Allocator.Persistent);
            m_MassesBuffer?.Release();
            m_MassesBuffer = new ComputeBuffer(m_ParticleCount, sizeof(float));

            m_Velocities.Dispose();
            m_Velocities = new NativeArray<float3>(m_ParticleCount, Allocator.Persistent);
            m_VelocitiesBuffer?.Release();
            m_VelocitiesBuffer = new ComputeBuffer(m_ParticleCount, Marshal.SizeOf(typeof(float3)));

            m_Particles.Dispose();
            m_Particles = new NativeArray<FluidParticleComponent>(m_ParticleCount, Allocator.Persistent);
            m_ParticlesBuffer?.Release();
            m_ParticlesBuffer = new ComputeBuffer(m_ParticleCount, Marshal.SizeOf(typeof(FluidParticleComponent)));

            m_Densities?.Release();
            m_Densities = new ComputeBuffer(m_ParticleCount, sizeof(float));

            m_Pressures?.Release();
            m_Pressures = new ComputeBuffer(m_ParticleCount, sizeof(float));

            changed = true;
        }
        if (m_BoundaryCount != m_BoundaryQuery.CalculateEntityCount())
        {
            m_BoundaryCount = m_BoundaryQuery.CalculateEntityCount();

            m_BoundaryPositions.Dispose();
            m_BoundaryPositions = new NativeArray<float3>(m_ParticleCount, Allocator.Persistent);
            m_BoundaryPositionsBuffer?.Release();
            m_BoundaryPositionsBuffer = new ComputeBuffer(m_ParticleCount, Marshal.SizeOf(typeof(float3)));

            m_BoundaryParticles.Dispose();
            m_BoundaryParticles = new NativeArray<BoundaryParticleComponent>(m_ParticleCount, Allocator.Persistent);
            m_BoundaryParticlesBuffer?.Release();
            m_BoundaryParticlesBuffer = new ComputeBuffer(m_ParticleCount, Marshal.SizeOf(typeof(BoundaryParticleComponent)));

            changed = true;
        }
        if (changed)
        {
            m_IndexMap?.Release();
            m_IndexMapCount = 1 << math.ceillog2(m_ParticleCount + m_BoundaryCount);
            m_IndexMap = new ComputeBuffer(m_IndexMapCount, sizeof(int) * 2);
        }

        int groups = (m_ParticleCount + m_BoundaryCount) / NUM_THREADS;
        if ((m_ParticleCount + m_BoundaryCount) % NUM_THREADS != 0) groups++;

        var positions = m_Positions;
        var masses = m_Masses;
        var velocities = m_Velocities;
        var particles = m_Particles;

        var collectParticalHandle = Entities
            .WithAll<FluidParticleComponent>()
            .ForEach((int entityInQueryIndex, in Translation translation, in PhysicsMass mass, in PhysicsVelocity velocity, in FluidParticleComponent particle) =>
            {
                positions[entityInQueryIndex] = translation.Value;
                masses[entityInQueryIndex] = mass.InverseMass;
                velocities[entityInQueryIndex] = velocity.Linear;
                particles[entityInQueryIndex] = particle;

                if (translation.Value.x < min[0]) min[0] = translation.Value.x;
                if (translation.Value.y < min[1]) min[1] = translation.Value.y;
                if (translation.Value.z < min[2]) min[2] = translation.Value.z;

                if (translation.Value.x > max[0]) max[0] = translation.Value.x;
                if (translation.Value.y > max[1]) max[1] = translation.Value.y;
                if (translation.Value.z > max[2]) max[2] = translation.Value.z;
            })
            .Schedule(Dependency);

        var boundaryPositions = m_BoundaryPositions;
        var boundaryParticles = m_BoundaryParticles;

        var collectBoundaryHandle = Entities
            .WithAll<BoundaryParticleComponent>()
            .ForEach((int entityInQueryIndex, in Translation translation, in BoundaryParticleComponent particle) =>
            {
                boundaryPositions[entityInQueryIndex] = translation.Value;
                boundaryParticles[entityInQueryIndex] = particle;

                if (translation.Value.x < min[0]) min[0] = translation.Value.x;
                if (translation.Value.y < min[1]) min[1] = translation.Value.y;
                if (translation.Value.z < min[2]) min[2] = translation.Value.z;

                if (translation.Value.x > max[0]) max[0] = translation.Value.x;
                if (translation.Value.y > max[1]) max[1] = translation.Value.y;
                if (translation.Value.z > max[2]) max[2] = translation.Value.z;
            })
            .Schedule(collectParticalHandle);

        Dependency = JobHandle.CombineDependencies(collectBoundaryHandle, findGridSizeJobHandle);
        Dependency.Complete();

        Vector3 gridLength = new Vector3(
            math.ceil((max[0] - min[0]) / gridSize[0]),
            math.ceil((max[1] - min[1]) / gridSize[0]),
            math.ceil((max[2] - min[2]) / gridSize[0])
        );
        if (m_TableCount != gridLength.x * gridLength.y * gridLength.z)
        {
            m_TableCount = (int)(gridLength.x * gridLength.y * gridLength.z);
            m_Table?.Release();
            m_Table = new ComputeBuffer(m_TableCount, sizeof(int) * 2);
        }

        m_PositionsBuffer.SetData(positions);
        m_MassesBuffer.SetData(masses);
        m_VelocitiesBuffer.SetData(velocities);
        m_ParticlesBuffer.SetData(particles);

        m_BoundaryPositionsBuffer.SetData(boundaryPositions);
        m_BoundaryParticlesBuffer.SetData(boundaryParticles);

        m_Shader.SetInt("ParticleCount", m_ParticleCount);
        m_Shader.SetInt("BoundaryCount", m_BoundaryCount);
        m_Shader.SetFloat("GridSize", gridSize[0]);
        m_Shader.SetVector("GridMin", new Vector4(min[0], min[1], min[2], 0));
        m_Shader.SetVector("GridLength", gridLength);
        m_Shader.SetFloat("KernelRadiusRate", FluidSPHUtils.KernelRadiusRate);

        HashPosition();
        BitonicSort();
        MapTable();
        ComputeDensityAndPressure();

        gridSize.Dispose(Dependency);
        min.Dispose(Dependency);
        max.Dispose(Dependency);
    }

    private void HashPosition()
    {
        var hashKernel = m_Shader.FindKernel("HashPosition");
        m_Shader.SetBuffer(hashKernel, "Positions", m_PositionsBuffer);
        m_Shader.SetBuffer(hashKernel, "BoundaryPositions", m_BoundaryPositionsBuffer);
        m_Shader.SetBuffer(hashKernel, "IndexMap", m_IndexMap);

        m_Shader.Dispatch(hashKernel, ComputeGroups(m_ParticleCount + m_BoundaryCount), 1, 1);
    }

    private void BitonicSort()
    {
        int groups = ComputeGroups(m_IndexMapCount / 2);
        var sortKernel = m_Shader.FindKernel("BitonicSort");
        m_Shader.SetInt("IndexMapCount", m_IndexMapCount);
        m_Shader.SetBuffer(sortKernel, "IndexMap", m_IndexMap);

        for (int h = 2; h <= m_IndexMapCount; h *= 2)
        {
            m_Shader.SetInt("BitonicStage", BITONIC_FLIP);
            m_Shader.SetInt("BitonicBlockSize", h);
            m_Shader.Dispatch(sortKernel, groups, 1, 1);

            for (int hh = h / 2; hh > 1; hh /= 2)
            {
                m_Shader.SetInt("BitonicStage", BITONIC_DISPERSE);
                m_Shader.SetInt("BitonicBlockSize", hh);
                m_Shader.Dispatch(sortKernel, groups, 1, 1);
            }
        }
    }

    private void MapTable()
    {
        var mapKernel = m_Shader.FindKernel("MapTable");

        m_Shader.SetBuffer(mapKernel, "Table", m_Table);
        m_Shader.SetBuffer(mapKernel, "IndexMap", m_IndexMap);
        m_Shader.Dispatch(mapKernel, ComputeGroups(m_TableCount), 1, 1);
    }

    private void ComputeDensityAndPressure()
    {
        var computeKernel = m_Shader.FindKernel("ComputeDensityAndPressure");

        m_Shader.SetBuffer(computeKernel, "Positions", m_PositionsBuffer);
        m_Shader.SetBuffer(computeKernel, "BoundaryPositions", m_BoundaryParticlesBuffer);
        m_Shader.SetBuffer(computeKernel, "Masses", m_MassesBuffer);
        m_Shader.SetBuffer(computeKernel, "Particles", m_ParticlesBuffer);
        m_Shader.SetBuffer(computeKernel, "Table", m_Table);
        m_Shader.SetBuffer(computeKernel, "IndexMap", m_IndexMap);
        m_Shader.SetBuffer(computeKernel, "Densities", m_Densities);
        m_Shader.SetBuffer(computeKernel, "Pressures", m_Pressures);
        m_Shader.Dispatch(computeKernel, ComputeGroups(m_ParticleCount), 1, 1);
    }

    private int ComputeGroups(int total)
    {
        int groups = total / NUM_THREADS;
        if (total % NUM_THREADS != 0) groups++;
        return groups;
    }

    private void DebugLogBuffer<T>(ComputeBuffer buffer, int start, int end) where T : struct
    {
        var arr = new NativeArray<T>(buffer.count, Allocator.Temp);
        var request = AsyncGPUReadback.RequestIntoNativeArray(ref arr, buffer);
        request.WaitForCompletion();
        for (int i = start; i < end; i++)
        {
            Debug.Log($"{i}, {arr[i]}");
        }
        arr.Dispose();
    }

    protected override void OnDestroy()
    {
        m_Positions.Dispose();
        m_PositionsBuffer?.Release();

        m_Masses.Dispose();
        m_MassesBuffer?.Release();

        m_Velocities.Dispose();
        m_VelocitiesBuffer?.Release();

        m_Particles.Dispose();
        m_ParticlesBuffer?.Release();

        m_BoundaryPositions.Dispose();
        m_BoundaryPositionsBuffer?.Release();

        m_BoundaryParticles.Dispose();
        m_BoundaryParticlesBuffer?.Release();

        m_IndexMap?.Release();
        m_Table?.Release();

        m_Densities?.Release();
        m_Pressures?.Release();
    }
}
