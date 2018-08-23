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

#ifndef vtk_m_worklet_KdTreeNNSearch_h
#define vtk_m_worklet_KdTreeNNSearch_h

#include <vtkm/cont/DeviceAdapterAlgorithm.h>

#include <vtkm/Math.h>
#include <vtkm/VectorAnalysis.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/ArrayHandleReverse.h>

#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/internal/DispatcherBase.h>
#include <vtkm/worklet/internal/WorkletBase.h>

#include <iostream>

namespace vtkm
{
namespace worklet
{
namespace spatialstructure
{

template < int N_DIMS >
class KdTreeNNSearch
{
public:
    class NearestNeighborSearchWorklet : public vtkm::worklet::WorkletMapField
    {
    public:
        using ControlSignature = void(FieldIn<> qcIn,
                                      WholeArrayIn<> treeIdIn,
                                      WholeArrayIn<> treeSplitIdIn,
                                      WholeArrayIn<> treeCoordiIn,
                                      FieldOut<> nnIdOut,
                                      FieldInOut<> nnDisOut);
        using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

        VTKM_CONT
        NearestNeighborSearchWorklet() {}

        template <typename CooriVecT, typename CooriT, typename IdPortalT, typename CoordiPortalT>
        VTKM_EXEC_CONT void NearestNeighborSearch(const CooriVecT& qc,
                CooriT& dis,
                vtkm::Id hopsSoFar,
                vtkm::Id& nnpIdx,
                vtkm::Int32 level,
                vtkm::Id sIdx,
                vtkm::Id tIdx,
                const IdPortalT& treePortal,
                const IdPortalT& splitIdPortal,
                const CoordiPortalT& coordiPortal) const
        {
            ++hopsSoFar;
            if( hopsSoFar > 10000 )
            {
                std::cerr << "over 10000 iterations for qc=" << qc[ 0 ] << "," << qc[ 1 ]  << std::endl;
                exit( 1 );
            } 

            if (tIdx - sIdx == 1)
            {
                ///// leaf node

                if( sIdx < 0 || sIdx >= treePortal.GetNumberOfValues() )
                {
                    std::cerr << "sIdx out of bounds in vtkm::kd" << std::endl;
                    exit( 1 );
                }

                vtkm::Id leafNodeIdx = treePortal.Get(sIdx);

                CooriVecT leaf;

                for( int i =0; i < N_DIMS; ++i )
                {
                    if( leafNodeIdx < 0 || leafNodeIdx >= coordiPortal.GetNumberOfValues() )
                    {
                        std::cerr << "leafNodeIdx out of bounds in vtkm::kd" << std::endl;
                        exit( 1 );
                    }
                    leaf[ i ] = coordiPortal.Get(leafNodeIdx)[ i ];
                }

                CooriT _dis = vtkm::Magnitude( leaf - qc );

                if (_dis < dis)
                {
                    dis = _dis;
                    nnpIdx = leafNodeIdx;
                }
            }
            else
            {
                //normal Node
                vtkm::Id splitNodeLoc = static_cast<vtkm::Id>(vtkm::Ceil(double((sIdx + tIdx)) / 2.0));

                if( splitNodeLoc < 0 || splitNodeLoc >= splitIdPortal.GetNumberOfValues() )
                {
                    std::cerr << "splitNodeLoc out of bounds in vtkm::kd" << std::endl;
                    exit( 1 );
                }

                if( splitIdPortal.Get(splitNodeLoc) < 0 || splitIdPortal.Get(splitNodeLoc) >= coordiPortal.GetNumberOfValues() )
                {
                    std::cerr << "splitIdPortal.Get(splitNodeLoc) out of bounds in vtkm::kd" << std::endl;
                    exit( 1 );
                }

                if( level % N_DIMS < 0 || level % N_DIMS >= 2 )
                {
                    std::cerr << "level mod N_DIMS out of bounds in vtkm::kd" << std::endl;
                    exit( 1 );
                }

                CooriT splitAxis = coordiPortal.Get(splitIdPortal.Get(splitNodeLoc))[ level % N_DIMS ];

                CooriT queryCoordi = qc[ level % N_DIMS ];
                ///

                if (queryCoordi <= splitAxis)
                {
                    //left tree first
                    if (queryCoordi - dis <= splitAxis)
                        NearestNeighborSearch(  qc,
                                                hopsSoFar,
                                                dis,
                                                nnpIdx,
                                                level + 1,
                                                sIdx,
                                                splitNodeLoc,
                                                treePortal,
                                                splitIdPortal,
                                                coordiPortal);
                    if (queryCoordi + dis > splitAxis)
                        NearestNeighborSearch(qc,
                                              hopsSoFar,                          
                                              dis,
                                              nnpIdx,
                                              level + 1,
                                              splitNodeLoc,
                                              tIdx,
                                              treePortal,
                                              splitIdPortal,
                                              coordiPortal);
                }
                else
                {
                    //right tree first
                    if (queryCoordi + dis > splitAxis)
                        NearestNeighborSearch(qc,
                                              hopsSoFar,
                                              dis,
                                              nnpIdx,
                                              level + 1,
                                              splitNodeLoc,
                                              tIdx,
                                              treePortal,
                                              splitIdPortal,
                                              coordiPortal);
                    if (queryCoordi - dis <= splitAxis)
                        NearestNeighborSearch(qc,
                                              hopsSoFar,
                                              dis,
                                              nnpIdx,
                                              level + 1,
                                              sIdx,
                                              splitNodeLoc,
                                              treePortal,
                                              splitIdPortal,
                                              coordiPortal);
                }
            }
        }

        template <typename CoordiVecType,
                  typename IdPortalType,
                  typename CoordiPortalType,
                  typename IdType,
                  typename CoordiType>
        VTKM_EXEC void operator()(const CoordiVecType& qc,
                                  const IdPortalType& treeIdPortal,
                                  const IdPortalType& treeSplitIdPortal,
                                  const CoordiPortalType& treeCoordiPortal,
                                  IdType& nnId,
                                  CoordiType& nnDis) const
        {
            vtkm::Id hopsSoFar = 0;
            NearestNeighborSearch(qc,
                                  hopsSoFar,
                                  nnDis,
                                  nnId,
                                  0,
                                  0,
                                  treeIdPortal.GetNumberOfValues(),
                                  treeIdPortal,
                                  treeSplitIdPortal,
                                  treeCoordiPortal);
        }
    };

/// \brief Execute the Neaseat Neighbor Search given kdtree and search points
///
/// Given x, y, z coordinate of of training data points in \c coordi_Handle, indices to KD-tree
/// leaf nodes in \c pointId_Handle and indices to internal nodes in \c splitId_Handle, search
/// for nearest neighbors in the training data points for each of testing points in \c qc_Handle.
/// Returns indices to nearest neighbor in \c nnId_Handle and distance to nearest neighbor in
/// \c nnDis_Handle.

    template <typename CoordType,
              typename CoordStorageTag1,
              typename CoordStorageTag2,
              typename DeviceAdapter>
    void Run(const vtkm::cont::ArrayHandle<vtkm::Vec<CoordType, N_DIMS>, CoordStorageTag1>& coordi_Handle,
             const vtkm::cont::ArrayHandle<vtkm::Id>& pointId_Handle,
             const vtkm::cont::ArrayHandle<vtkm::Id>& splitId_Handle,
             const vtkm::cont::ArrayHandle<vtkm::Vec<CoordType, N_DIMS>, CoordStorageTag2>& qc_Handle,
             vtkm::cont::ArrayHandle<vtkm::Id>& nnId_Handle,
             vtkm::cont::ArrayHandle<CoordType>& nnDis_Handle,
             DeviceAdapter)
    {
        //fill the nnDis_Handle handle array with max values before running
        auto intialValue = std::numeric_limits<CoordType>::max();
        vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy(
            vtkm::cont::make_ArrayHandleConstant(intialValue, qc_Handle.GetNumberOfValues()),
            nnDis_Handle);

        // set up stack size for cuda environment
#ifdef VTKM_CUDA
        using DeviceAdapterTraits = vtkm::cont::DeviceAdapterTraits<DeviceAdapter>;
    std::size_t stackSizeBackup;
    (void)stackSizeBackup;
        if (DeviceAdapterTraits::GetId() == VTKM_DEVICE_ADAPTER_CUDA)
        {
            cudaDeviceGetLimit(&stackSizeBackup, cudaLimitStackSize);

            cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 32);
            std::cout << "set stack size = " << 1024 * 32 << std::endl;
        }
#endif

        NearestNeighborSearchWorklet nns3dWorklet;
        vtkm::worklet::DispatcherMapField<NearestNeighborSearchWorklet, DeviceAdapter>
        nnsDispatcher(nns3dWorklet);
        nnsDispatcher.Invoke(
            qc_Handle, pointId_Handle, splitId_Handle, coordi_Handle, nnId_Handle, nnDis_Handle);

#ifdef VTKM_CUDA
        if (DeviceAdapterTraits::GetId() == VTKM_DEVICE_ADAPTER_CUDA)
        {
            cudaDeviceSetLimit(cudaLimitStackSize, stackSizeBackup);
            std::cout << "set stack size back to default= " << stackSizeBackup << std::endl;
        }
#endif
    }
};
}
}
} // namespace vtkm::worklet

#endif // vtk_m_worklet_KdTreeNNSearch_h
