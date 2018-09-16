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
        VTKM_EXEC_CONT void NearestNeighborSearchIterative(
            const CooriVecT & qc,
            CooriT & dis,
            vtkm::Id & nnpIdx,
            vtkm::Int32 N,
            const IdPortalT & treePortal,
            const IdPortalT & splitIdPortal,
            const CoordiPortalT & coordiPortal ) const
        {
            const vtkm::Int32 MAX_STACK_SIZE = 30000;
            
            vtkm::Int32 stacki[ MAX_STACK_SIZE ];
            CooriT      stackf[ MAX_STACK_SIZE ];
            vtkm::UInt8 stackc[ MAX_STACK_SIZE ];

            stacki[ 0 ] = 0;
            stacki[ 1 ] = N;

            stackc[ 0 ] = 0;
            stackc[ 1 ] = 0;

            stackf[ 0 ] = 0;
            stackf[ 1 ] = 0;

            vtkm::Int32 stackSize = 1;

            while( stackSize > 0 )
            {   
                vtkm::Int32 left  = stacki[ ( stackSize - 1 )*2     ];
                vtkm::Int32 right = stacki[ ( stackSize - 1 )*2 + 1 ];
                
                vtkm::UInt8 dim = stackc[ ( stackSize - 1 )*2     ];                
                vtkm::UInt8 cond = stackc[ ( stackSize -  1 )*2 + 1 ];

                CooriT qcrdi = stackf[ ( stackSize - 1 )*2     ];
                CooriT splta = stackf[ ( stackSize - 1 )*2 + 1 ];
                                                
                --stackSize;

                if( ( qcrdi - dis  > splta && cond == 0 )
                 || ( qcrdi + dis <= splta && cond == 1 ) )
                {
                    continue;
                }

                if ( right - left == 1 )
                {
                    ///// leaf node
                    const vtkm::Id & leafNodeIdx = treePortal.Get( left );
                    const CooriT _dis = vtkm::Magnitude( coordiPortal.Get( leafNodeIdx ) - qc );
                    
                    if ( _dis < dis )
                    {
                        dis = _dis;
                        nnpIdx = leafNodeIdx;
                    }                   
                }
                else
                {
                    //normal Node
                    // const vtkm::Int32 DIM_INDEX = dim % N_DIMS;
                    vtkm::Int32 splitNodeLoc = static_cast< vtkm::Int32 >( vtkm::Ceil( double( ( left + right ) ) / 2.0 ) );
                    CooriT splitAxis = coordiPortal.Get( splitIdPortal.Get( splitNodeLoc ) )[ dim ];
                    CooriT queryCoordi = qc[ dim ];
                   
                   const vtkm::UInt8 nextDim = ( dim + 1 ) % N_DIMS;

                    if ( queryCoordi <= splitAxis )
                    { 
                        if ( queryCoordi + dis > splitAxis )
                        {
                            ++stackSize;                           
                            stacki[ ( stackSize - 1 )*2     ] = splitNodeLoc;
                            stacki[ ( stackSize - 1 )*2 + 1 ] = right;

                            stackc[ ( stackSize - 1 )*2     ] = nextDim;  
                            stackc[ ( stackSize - 1 )*2 + 1 ] = 1;  
                            
                            stackf[ ( stackSize - 1 )*2     ] = queryCoordi;  
                            stackf[ ( stackSize - 1 )*2 + 1 ] = splitAxis;            
                        }

                        // left tree first
                        if ( queryCoordi - dis <= splitAxis )
                        {
                            ++stackSize;
                            stacki[ ( stackSize - 1 )*2     ] = left;
                            stacki[ ( stackSize - 1 )*2 + 1 ] = splitNodeLoc;

                            stackc[ ( stackSize - 1 )*2     ] = nextDim;                               
                            stackc[ ( stackSize - 1 )*2 + 1 ] = 0;   

                            stackf[ ( stackSize - 1 )*2     ] = queryCoordi;  
                            stackf[ ( stackSize - 1 )*2 + 1 ] = splitAxis;                                                                             
                        }   
                    }
                    else
                    {
                        if ( queryCoordi - dis <= splitAxis )
                        {
                            ++stackSize;
                            stacki[ ( stackSize - 1 )*2     ] = left;
                            stacki[ ( stackSize - 1 )*2 + 1 ] = splitNodeLoc;

                            stackc[ ( stackSize - 1 )*2     ] = nextDim;                               
                            stackc[ ( stackSize - 1 )*2 + 1 ] = 0;   

                            stackf[ ( stackSize - 1 )*2     ] = queryCoordi;  
                            stackf[ ( stackSize - 1 )*2 + 1 ] = splitAxis;                                  
                        }

                        // right tree first
                        if ( queryCoordi + dis > splitAxis )
                        {
                            ++stackSize;                           
                            stacki[ ( stackSize - 1 )*2     ] = splitNodeLoc;
                            stacki[ ( stackSize - 1 )*2 + 1 ] = right;

                            stackc[ ( stackSize - 1 )*2     ] = nextDim;  
                            stackc[ ( stackSize - 1 )*2 + 1 ] = 1;  
                            
                            stackf[ ( stackSize - 1 )*2     ] = queryCoordi;  
                            stackf[ ( stackSize - 1 )*2 + 1 ] = splitAxis;                                      
                        }       
                    }
                }
            }
        }

        template <typename CooriVecT, typename CooriT, typename IdPortalT, typename CoordiPortalT>
        VTKM_EXEC_CONT void NearestNeighborSearch(
            const CooriVecT & qc,
            CooriT & dis,
            vtkm::Id & nnpIdx,
            vtkm::Int32 level,
            vtkm::Int32 left,
            vtkm::Int32 right,
            const IdPortalT     & treePortal,
            const IdPortalT     & splitIdPortal,
            const CoordiPortalT & coordiPortal ) const
        {
            if ( right - left == 1 )
            {
                ///// leaf node
                const vtkm::Id & leafNodeIdx = treePortal.Get( left );
                const CooriT _dis = vtkm::Magnitude( coordiPortal.Get( leafNodeIdx ) - qc );
                if ( _dis < dis )
                {
                    dis = _dis;
                    nnpIdx = leafNodeIdx;
                }    
            }
            else
            {
                //normal Node
                const vtkm::Int32 DIM_INDEX = level % N_DIMS;
                vtkm::Int32 splitNodeLoc = static_cast< vtkm::Int32 >( vtkm::Ceil( double( ( left + right ) ) / 2.0 ) );
                CooriT splitAxis = coordiPortal.Get( splitIdPortal.Get( splitNodeLoc ) )[ DIM_INDEX ];
                CooriT queryCoordi = qc[ DIM_INDEX ];

                if ( queryCoordi <= splitAxis )
                { 
                    //left tree first
                    if ( queryCoordi - dis <= splitAxis )
                    {
                        NearestNeighborSearch(
                            qc,
                            dis,
                            nnpIdx,
                            level + 1,
                            left,
                            splitNodeLoc,
                            treePortal,
                            splitIdPortal,
                            coordiPortal );
                    }
                    if ( queryCoordi + dis > splitAxis )
                    {
                        NearestNeighborSearch(
                            qc,                        
                            dis,
                            nnpIdx,
                            level + 1,
                            splitNodeLoc,
                            right,
                            treePortal,
                            splitIdPortal,
                            coordiPortal);
                    }
                }
                else
                {
                    //right tree first
                    if ( queryCoordi + dis > splitAxis )
                    {
                        NearestNeighborSearch(
                            qc,
                            dis,
                            nnpIdx,
                            level + 1,
                            splitNodeLoc,
                            right,
                            treePortal,
                            splitIdPortal,
                            coordiPortal);
                    }
                    if ( queryCoordi - dis <= splitAxis )
                    {
                        NearestNeighborSearch(
                            qc,
                            dis,
                            nnpIdx,
                            level + 1,
                            left,
                            splitNodeLoc,
                            treePortal,
                            splitIdPortal,
                            coordiPortal);
                    }
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
            // NearestNeighborSearch(
            //     qc,
            //     nnDis,
            //     nnId,
            //     0,
            //     0,
            //     treeIdPortal.GetNumberOfValues(),
            //     treeIdPortal,
            //     treeSplitIdPortal,
            //     treeCoordiPortal);

            NearestNeighborSearchIterative(
                qc,
                nnDis,
                nnId,
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
            nnDis_Handle );

        // set up stack size for cuda environment
#ifdef VTKM_CUDA
        using DeviceAdapterTraits = vtkm::cont::DeviceAdapterTraits<DeviceAdapter>;
    std::size_t stackSizeBackup;
    (void)stackSizeBackup;
        if (DeviceAdapterTraits::GetId() == VTKM_DEVICE_ADAPTER_CUDA)
        {
            cudaDeviceGetLimit(&stackSizeBackup, cudaLimitStackSize);

            cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 32);
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
        }
#endif
    }
};
}
}
} // namespace vtkm::worklet

#endif // vtk_m_worklet_KdTreeNNSearch_h
