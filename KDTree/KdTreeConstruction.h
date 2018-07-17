//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014 Los Alamos National Security.
//
//  Under the terms of Contract DE-NA0003525 with NTESS,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#ifndef vtk_m_worklet_KdTreeConstruction_h
#define vtk_m_worklet_KdTreeConstruction_h

#include <vtkm/Math.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/ArrayHandleReverse.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/cont/arg/ControlSignatureTagBase.h>
#include <vtkm/cont/serial/DeviceAdapterSerial.h>
#include <vtkm/cont/testing/Testing.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/internal/DispatcherBase.h>
#include <vtkm/worklet/internal/WorkletBase.h>

#include <array>

namespace vtkm
{
namespace worklet
{
namespace spatialstructure
{

template <int N_DIMS>
class KdTreeConstruction
{
public:
    ////////// General WORKLET for Kd-tree  //////
    class ComputeFlag : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> rank, FieldIn<> pointCountInSeg, FieldOut<> flag);
        using ExecutionSignature = void(_1, _2, _3);

        VTKM_CONT
        ComputeFlag() {}

        template <typename T>
        VTKM_EXEC void operator()(const T& rank, const T& pointCountInSeg, T& flag) const
        {
            if (static_cast<float>(rank) >= static_cast<float>(pointCountInSeg) / 2.0f)
                flag = 1; //right subtree
            else
                flag = 0; //left subtree
        }
    };

    class InverseArray : public vtkm::worklet::WorkletMapField
    {
        //only for 0/1 array
    public:
        using ControlSignature = void(FieldIn<> in, FieldOut<> out);
        using ExecutionSignature = void(_1, _2);

        VTKM_CONT
        InverseArray() {}

        template <typename T>
        VTKM_EXEC void operator()(const T& in, T& out) const
        {
            if (in == 0)
                out = 1;
            else
                out = 0;
        }
    };

    class SegmentedSplitTransform : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature =
            void(FieldIn<> B, FieldIn<> D, FieldIn<> F, FieldIn<> G, FieldIn<> H, FieldOut<> I);
        using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

        VTKM_CONT
        SegmentedSplitTransform() {}

        template <typename T>
        VTKM_EXEC void operator()(const T& B, const T& D, const T& F, const T& G, const T& H, T& I)
        const
        {
            if (B == 1)
            {
                I = F + H + D;
            }
            else
            {
                I = F + G - 1;
            }
        }
    };

    class ScatterArray : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> in, FieldIn<> index, WholeArrayOut<> out);
        using ExecutionSignature = void(_1, _2, _3);

        VTKM_CONT
        ScatterArray() {}

        template <typename T, typename OutputArrayPortalType>
        VTKM_EXEC void operator()(const T& in,
                                  const T& index,
                                  const OutputArrayPortalType& outputPortal) const
        {
            outputPortal.Set(index, in);
        }
    };

    class NewSegmentId : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> inSegmentId, FieldIn<> flag, FieldOut<> outSegmentId);
        using ExecutionSignature = void(_1, _2, _3);

        VTKM_CONT
        NewSegmentId() {}

        template <typename T>
        VTKM_EXEC void operator()(const T& oldSegId, const T& flag, T& newSegId) const
        {
            if (flag == 0)
                newSegId = oldSegId * 2;
            else
                newSegId = oldSegId * 2 + 1;
        }
    };

    class SaveSplitPointId : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> pointId,
                                      FieldIn<> flag,
                                      FieldIn<> oldSplitPointId,
                                      FieldOut<> newSplitPointId);
        using ExecutionSignature = void(_1, _2, _3, _4);

        VTKM_CONT
        SaveSplitPointId() {}

        template <typename T>
        VTKM_EXEC void operator()(const T& pointId,
                                  const T& flag,
                                  const T& oldSplitPointId,
                                  T& newSplitPointId) const
        {
            if (flag == 0) //do not change
                newSplitPointId = oldSplitPointId;
            else //split point id
                newSplitPointId = pointId;
        }
    };

    class FindSplitPointId : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> pointId, FieldIn<> rank, FieldOut<> splitIdInsegment);
        using ExecutionSignature = void(_1, _2, _3);

        VTKM_CONT
        FindSplitPointId() {}

        template <typename T>
        VTKM_EXEC void operator()(const T& pointId, const T& rank, T& splitIdInsegment) const
        {
            if (rank == 0) //do not change
                splitIdInsegment = pointId;
            else                     //split point id
                splitIdInsegment = -1; //indicate this is not split point
        }
    };

    class ArrayAdd : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> inArray0, FieldIn<> inArray1, FieldOut<> outArray);
        using ExecutionSignature = void(_1, _2, _3);

        VTKM_CONT
        ArrayAdd() {}

        template <typename T>
        VTKM_EXEC void operator()(const T& in0, const T& in1, T& out) const
        {
            out = in0 + in1;
        }
    };

    class SeprateVecElementAryHandle : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> inVec,
                                      FieldOut<> out );
        using ExecutionSignature = void(_1, _2 );

        int m_dim;

        VTKM_CONT
        SeprateVecElementAryHandle() {}

        void setDim( int n )
        {
            m_dim = n;
        }

        template <typename T>
        VTKM_EXEC void operator()(const Vec<T, N_DIMS>& inVec, T& out ) const
        {
            out = inVec[ m_dim ];
        }
    };

    ////////// General worklet WRAPPER for Kd-tree //////
    template <typename T, class BinaryFunctor, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> ReverseScanInclusiveByKey(vtkm::cont::ArrayHandle<T>& keyHandle,
            vtkm::cont::ArrayHandle<T>& dataHandle,
            BinaryFunctor binary_functor,
            DeviceAdapter vtkmNotUsed(device))
    {
        using Algorithm = typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

        vtkm::cont::ArrayHandle<T> resultHandle;

        auto reversedResultHandle = vtkm::cont::make_ArrayHandleReverse(resultHandle);

        Algorithm::ScanInclusiveByKey(vtkm::cont::make_ArrayHandleReverse(keyHandle),
                                      vtkm::cont::make_ArrayHandleReverse(dataHandle),
                                      reversedResultHandle,
                                      binary_functor);

        return resultHandle;
    }

    template <typename T, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> Inverse01ArrayWrapper(vtkm::cont::ArrayHandle<T>& inputHandle,
            DeviceAdapter vtkmNotUsed(device))
    {
        vtkm::cont::ArrayHandle<T> InverseHandle;
        InverseArray invWorklet;
        vtkm::worklet::DispatcherMapField<InverseArray, DeviceAdapter> InverseArrayDispatcher(
            invWorklet);
        InverseArrayDispatcher.Invoke(inputHandle, InverseHandle);
        return InverseHandle;
    }

    template <typename T, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> ScatterArrayWrapper(vtkm::cont::ArrayHandle<T>& inputHandle,
            vtkm::cont::ArrayHandle<T>& indexHandle,
            DeviceAdapter vtkmNotUsed(device))
    {
        vtkm::cont::ArrayHandle<T> outputHandle;
        outputHandle.Allocate(inputHandle.GetNumberOfValues());
        ScatterArray scatterWorklet;
        vtkm::worklet::DispatcherMapField<ScatterArray, DeviceAdapter> ScatterArrayDispatcher(
            scatterWorklet);
        ScatterArrayDispatcher.Invoke(inputHandle, indexHandle, outputHandle);
        return outputHandle;
    }

    template <typename T, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> NewKeyWrapper(vtkm::cont::ArrayHandle<T>& oldSegIdHandle,
            vtkm::cont::ArrayHandle<T>& flagHandle,
            DeviceAdapter vtkmNotUsed(device))
    {
        vtkm::cont::ArrayHandle<T> newSegIdHandle;
        NewSegmentId newsegidWorklet;
        vtkm::worklet::DispatcherMapField<NewSegmentId, DeviceAdapter> newSegIdDispatcher(
            newsegidWorklet);
        newSegIdDispatcher.Invoke(oldSegIdHandle, flagHandle, newSegIdHandle);
        return newSegIdHandle;
    }

    template <typename T, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> SaveSplitPointIdWrapper(vtkm::cont::ArrayHandle<T>& pointIdHandle,
            vtkm::cont::ArrayHandle<T>& flagHandle,
            vtkm::cont::ArrayHandle<T>& rankHandle,
            vtkm::cont::ArrayHandle<T>& oldSplitIdHandle,
            DeviceAdapter device)
    {
        vtkm::cont::ArrayHandle<T> splitIdInSegmentHandle;
        FindSplitPointId findSplitPointIdWorklet;
        vtkm::worklet::DispatcherMapField<FindSplitPointId, DeviceAdapter>
        findSplitPointIdWorkletDispatcher(findSplitPointIdWorklet);
        findSplitPointIdWorkletDispatcher.Invoke(pointIdHandle, rankHandle, splitIdInSegmentHandle);

        vtkm::cont::ArrayHandle<T> splitIdInSegmentByScanHandle =
            ReverseScanInclusiveByKey(flagHandle, splitIdInSegmentHandle, vtkm::Maximum(), device);

        vtkm::cont::ArrayHandle<T> splitIdHandle;
        SaveSplitPointId saveSplitPointIdWorklet;
        vtkm::worklet::DispatcherMapField<SaveSplitPointId, DeviceAdapter>
        saveSplitPointIdWorkletDispatcher(saveSplitPointIdWorklet);
        saveSplitPointIdWorkletDispatcher.Invoke(
            splitIdInSegmentByScanHandle, flagHandle, oldSplitIdHandle, splitIdHandle);

        return splitIdHandle;
    }

    template <typename T, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> ArrayAddWrapper(vtkm::cont::ArrayHandle<T>& array0Handle,
            vtkm::cont::ArrayHandle<T>& array1Handle,
            DeviceAdapter vtkmNotUsed(device))
    {
        vtkm::cont::ArrayHandle<T> resultHandle;
        ArrayAdd arrayAddWorklet;
        vtkm::worklet::DispatcherMapField<ArrayAdd, DeviceAdapter> arrayAddDispatcher(arrayAddWorklet);
        arrayAddDispatcher.Invoke(array0Handle, array1Handle, resultHandle);
        return resultHandle;
    }

    ///////////////////////////////////////////////////
    ////////General Kd tree function //////////////////
    ///////////////////////////////////////////////////
    template <typename T, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> ComputeFlagProcedure(vtkm::cont::ArrayHandle<T>& rankHandle,
            vtkm::cont::ArrayHandle<T>& segIdHandle,
            DeviceAdapter device)
    {
        using Algorithm = typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

        vtkm::cont::ArrayHandle<T> segCountAryHandle;
        {
            vtkm::cont::ArrayHandle<T> tmpAryHandle;
            vtkm::cont::ArrayHandleConstant<T> constHandle(1, rankHandle.GetNumberOfValues());
            Algorithm::ScanInclusiveByKey(
                segIdHandle, constHandle, tmpAryHandle, vtkm::Add()); //compute ttl segs in segment

            segCountAryHandle =
                ReverseScanInclusiveByKey(segIdHandle, tmpAryHandle, vtkm::Maximum(), device);
        }

        vtkm::cont::ArrayHandle<T> flagHandle;
        vtkm::worklet::DispatcherMapField<ComputeFlag, DeviceAdapter> ComputeFlagDispatcher;
        ComputeFlagDispatcher.Invoke(rankHandle, segCountAryHandle, flagHandle);

        return flagHandle;
    }

    template <typename T, typename DeviceAdapter>
    vtkm::cont::ArrayHandle<T> SegmentedSplitProcedure(vtkm::cont::ArrayHandle<T>& A_Handle,
            vtkm::cont::ArrayHandle<T>& B_Handle,
            vtkm::cont::ArrayHandle<T>& C_Handle,
            DeviceAdapter device)
    {
        using Algorithm = typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

        vtkm::cont::ArrayHandle<T> D_Handle;
        T initValue = 0;
        Algorithm::ScanExclusiveByKey(C_Handle, B_Handle, D_Handle, initValue, vtkm::Add());

        vtkm::cont::ArrayHandleCounting<T> Ecouting_Handle(0, 1, A_Handle.GetNumberOfValues());
        vtkm::cont::ArrayHandle<T> E_Handle;
        Algorithm::Copy(Ecouting_Handle, E_Handle);

        vtkm::cont::ArrayHandle<T> F_Handle;
        Algorithm::ScanInclusiveByKey(C_Handle, E_Handle, F_Handle, vtkm::Minimum());

        vtkm::cont::ArrayHandle<T> InvB_Handle = Inverse01ArrayWrapper(B_Handle, device);
        vtkm::cont::ArrayHandle<T> G_Handle;
        Algorithm::ScanInclusiveByKey(C_Handle, InvB_Handle, G_Handle, vtkm::Add());

        vtkm::cont::ArrayHandle<T> H_Handle =
            ReverseScanInclusiveByKey(C_Handle, G_Handle, vtkm::Maximum(), device);

        vtkm::cont::ArrayHandle<T> I_Handle;
        SegmentedSplitTransform sstWorklet;
        vtkm::worklet::DispatcherMapField<SegmentedSplitTransform, DeviceAdapter>
        SegmentedSplitTransformDispatcher(sstWorklet);
        SegmentedSplitTransformDispatcher.Invoke(
            B_Handle, D_Handle, F_Handle, G_Handle, H_Handle, I_Handle);

        return ScatterArrayWrapper(A_Handle, I_Handle, device);
    }

    template <typename T, typename DeviceAdapter>
    void RenumberRanksProcedure(vtkm::cont::ArrayHandle<T>& A_Handle,
                                vtkm::cont::ArrayHandle<T>& B_Handle,
                                vtkm::cont::ArrayHandle<T>& C_Handle,
                                vtkm::cont::ArrayHandle<T>& D_Handle,
                                DeviceAdapter device)
    {
        using Algorithm = typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

        vtkm::Id nPoints = A_Handle.GetNumberOfValues();

        vtkm::cont::ArrayHandleCounting<T> Ecouting_Handle(0, 1, nPoints);
        vtkm::cont::ArrayHandle<T> E_Handle;
        Algorithm::Copy(Ecouting_Handle, E_Handle);

        vtkm::cont::ArrayHandle<T> F_Handle;
        Algorithm::ScanInclusiveByKey(D_Handle, E_Handle, F_Handle, vtkm::Minimum());

        vtkm::cont::ArrayHandle<T> G_Handle;
        G_Handle = ArrayAddWrapper(A_Handle, F_Handle, device);

        vtkm::cont::ArrayHandleConstant<T> HConstant_Handle(1, nPoints);
        vtkm::cont::ArrayHandle<T> H_Handle;
        Algorithm::Copy(HConstant_Handle, H_Handle);

        vtkm::cont::ArrayHandle<T> I_Handle;
        T initValue = 0;
        Algorithm::ScanExclusiveByKey(C_Handle, H_Handle, I_Handle, initValue, vtkm::Add());

        vtkm::cont::ArrayHandle<T> J_Handle;
        J_Handle = ScatterArrayWrapper(I_Handle, G_Handle, device);

        vtkm::cont::ArrayHandle<T> K_Handle;
        K_Handle = ScatterArrayWrapper(B_Handle, G_Handle, device);

        vtkm::cont::ArrayHandle<T> L_Handle;
        L_Handle = SegmentedSplitProcedure(J_Handle, K_Handle, D_Handle, device);

        vtkm::cont::ArrayHandle<T> M_Handle;
        Algorithm::ScanInclusiveByKey(C_Handle, E_Handle, M_Handle, vtkm::Minimum());

        vtkm::cont::ArrayHandle<T> N_Handle;
        N_Handle = ArrayAddWrapper(L_Handle, M_Handle, device);

        A_Handle = ScatterArrayWrapper(I_Handle, N_Handle, device);
    }

    ///////////// construction      /////////////////////
    /// \brief Segmented split for  x, y, z coordinates
    ///
    /// Split \c pointId_Handle, \c X_Handle, \c Y_Handle and \c Z_Handle within each segment
    /// as indicated by \c segId_Handle according to flags in \c flag_Handle.
    ///
    /// \tparam T
    /// \tparam DeviceAdapter
    /// \param pointId_Handle
    /// \param flag_Handle
    /// \param segId_Handle
    /// \param X_Handle
    /// \param Y_Handle
    /// \param Z_Handle
    /// \param device
    template <typename T, typename DeviceAdapter>
    void SegmentedSplitProcedure(vtkm::cont::ArrayHandle<T>& pointId_Handle,
                                 vtkm::cont::ArrayHandle<T>& flag_Handle,
                                 vtkm::cont::ArrayHandle<T>& segId_Handle,
                                 std::array< vtkm::cont::ArrayHandle<T>, N_DIMS > & handles,
                                 DeviceAdapter device)
    {
        using Algorithm = typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

        vtkm::cont::ArrayHandle<T> D_Handle;
        T initValue = 0;
        Algorithm::ScanExclusiveByKey(segId_Handle, flag_Handle, D_Handle, initValue, vtkm::Add());

        vtkm::cont::ArrayHandleCounting<T> Ecouting_Handle(0, 1, pointId_Handle.GetNumberOfValues());
        vtkm::cont::ArrayHandle<T> E_Handle;
        Algorithm::Copy(Ecouting_Handle, E_Handle);

        vtkm::cont::ArrayHandle<T> F_Handle;
        Algorithm::ScanInclusiveByKey(segId_Handle, E_Handle, F_Handle, vtkm::Minimum());

        vtkm::cont::ArrayHandle<T> InvB_Handle = Inverse01ArrayWrapper(flag_Handle, device);
        vtkm::cont::ArrayHandle<T> G_Handle;
        Algorithm::ScanInclusiveByKey(segId_Handle, InvB_Handle, G_Handle, vtkm::Add());

        vtkm::cont::ArrayHandle<T> H_Handle =
            ReverseScanInclusiveByKey(segId_Handle, G_Handle, vtkm::Maximum(), device);

        vtkm::cont::ArrayHandle<T> I_Handle;

        SegmentedSplitTransform sstWorklet;

        vtkm::worklet::DispatcherMapField<SegmentedSplitTransform, DeviceAdapter>
        SegmentedSplitTransformDispatcher(sstWorklet);

        SegmentedSplitTransformDispatcher.Invoke(
            flag_Handle, D_Handle, F_Handle, G_Handle, H_Handle, I_Handle);

        pointId_Handle = ScatterArrayWrapper(pointId_Handle, I_Handle, device);

        flag_Handle = ScatterArrayWrapper(flag_Handle, I_Handle, device);

        for( int i = 0; i < N_DIMS; ++i )
        {
            handles[ i ] = ScatterArrayWrapper( handles[ i ], I_Handle, device);
        }
    }

    /// \brief Perform one level of KD-Tree construction
    ///
    /// Construct a level of KD-Tree by segemeted splits (partitioning) of \c pointId_Handle,
    /// \c xrank_Handle, \c yrank_Handle and \c zrank_Handle according to the medium element
    /// in each segment as indicated by \c segId_Handle alone the axis determined by \c level.
    /// The split point of each segment will be updated in \c splitId_Handle.
    template <typename T, typename DeviceAdapter>
    void OneLevelSplit(  vtkm::cont::ArrayHandle<T> & pointId_Handle,
                         std::array< vtkm::cont::ArrayHandle<T>, N_DIMS > & rank_Handles,
                         vtkm::cont::ArrayHandle<T> & segId_Handle,
                         vtkm::cont::ArrayHandle<T> & splitId_Handle,
                         vtkm::Int32 level,
                         DeviceAdapter device)
    {
        using Algorithm = typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

        vtkm::cont::ArrayHandle<T> flag_Handle = ComputeFlagProcedure( rank_Handles[ level % N_DIMS ], segId_Handle, device);

        SegmentedSplitProcedure(
            pointId_Handle, flag_Handle, segId_Handle, rank_Handles, device);

        vtkm::cont::ArrayHandle<T> segIdOld_Handle;
        Algorithm::Copy(segId_Handle, segIdOld_Handle);
        segId_Handle = NewKeyWrapper(segIdOld_Handle, flag_Handle, device);

        for( int i = 0; i < N_DIMS; ++i )
        {
            RenumberRanksProcedure( rank_Handles[ i ], flag_Handle, segId_Handle, segIdOld_Handle, device);
        }

        splitId_Handle =
            SaveSplitPointIdWrapper(pointId_Handle, flag_Handle, rank_Handles[ level % N_DIMS ], splitId_Handle, device);
    }

    /// \brief Construct KdTree from x y z coordinate vector.
    ///
    /// This method constructs an array based KD-Tree from x, y, z coordinates of points in \c
    /// coordi_Handle. The method rotates between x, y and z axis and splits input points into
    /// equal halves with respect to the split axis at each level of construction. The indices to
    /// the leaf nodes are returned in \c pointId_Handle and indices to internal nodes (splits)
    /// are returned in splitId_handle.
    ///
    /// \param coordi_Handle (in) x, y, z coordinates of input points
    /// \param pointId_Handle (out) returns indices to leaf nodes of the KD-tree
    /// \param splitId_Handle (out) returns indices to internal nodes of the KD-tree
    /// \param device the device to run the construction on
    // Leaf Node vector and internal node (split) vectpr
    template <typename CoordType, typename CoordStorageTag, typename DeviceAdapter>
    void Run(const vtkm::cont::ArrayHandle<vtkm::Vec<CoordType, N_DIMS>, CoordStorageTag>& coordi_Handle,
             vtkm::cont::ArrayHandle<vtkm::Id>& pointId_Handle,
             vtkm::cont::ArrayHandle<vtkm::Id>& splitId_Handle,
             DeviceAdapter device)
    {
        using Algorithm = typename vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

        vtkm::Id nTrainingPoints = coordi_Handle.GetNumberOfValues();
        vtkm::cont::ArrayHandleCounting<vtkm::Id> counting_Handle(0, 1, nTrainingPoints);
        vtkm::cont::ArrayHandle<vtkm::Id> order_Handles[ N_DIMS ];

        Algorithm::Copy(counting_Handle, pointId_Handle);

        for( int i = 0; i < N_DIMS; ++i )
        {
            Algorithm::Copy( counting_Handle, order_Handles[ i ] );
        }

        splitId_Handle.Allocate(nTrainingPoints);

        std::array< vtkm::cont::ArrayHandle<CoordType>, N_DIMS > coordi_Handles;
        std::array< SeprateVecElementAryHandle, N_DIMS > sepVecWorklets;
        for( int i = 0; i < N_DIMS; ++i )
        {
            sepVecWorklets[ i ].setDim( i );
        }
        std::array< vtkm::cont::ArrayHandle<vtkm::Id>, N_DIMS > rank_Handles;

        for( int i = 0; i < N_DIMS; ++i )
        {
            vtkm::worklet::DispatcherMapField<SeprateVecElementAryHandle, DeviceAdapter> sepVecispatcher(
                sepVecWorklets[ i ] );
            sepVecispatcher.Invoke(coordi_Handle, coordi_Handles[ i ] );

            Algorithm::SortByKey( coordi_Handles[ i ], order_Handles[ i ] );

            rank_Handles[ i ] = ScatterArrayWrapper( pointId_Handle, order_Handles[ i ], device);
        }

        vtkm::cont::ArrayHandle<vtkm::Id> segId_Handle;
        vtkm::cont::ArrayHandleConstant<vtkm::Id> constHandle(0, nTrainingPoints);
        Algorithm::Copy(constHandle, segId_Handle);

        ///// build kd tree /////
        vtkm::Int32 maxLevel = static_cast<vtkm::Int32>(ceil(vtkm::Log2(nTrainingPoints) + 1));
        for (vtkm::Int32 i = 0; i < maxLevel - 1; i++)
        {
            OneLevelSplit(pointId_Handle,
                          rank_Handles,
                          segId_Handle,
                          splitId_Handle,
                          i,
                          device );
        }
    }
};
}
}
} // namespace vtkm::worklet

#endif // vtk_m_worklet_KdTreeConstruction_h
