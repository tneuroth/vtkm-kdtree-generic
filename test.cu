#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/cont/testing/Testing.h>
#include "KDTree/KdTree.h"

#include <iostream>
#include <chrono>

using namespace std;

#define N_DIMS 2 

namespace
{

using Algorithm = vtkm::cont::Algorithm;

// template< typename DeviceAdapter >
// inline void checkDevice( DeviceAdapter)
// {
//     // using DeviceAdapterTraits = vtkm::cont::DeviceAdapterTraits<DeviceAdapter>;
//     // std::cout << "vtkm is using " << DeviceAdapterTraits::GetName() << std::endl;
// }


// template <typename CoordiVecT, typename CoordiPortalT, typename CoordiT>
// VTKM_EXEC_CONT vtkm::Id NNSVerify3D(CoordiVecT qc, CoordiPortalT coordiPortal, CoordiT& dis)

////brute force method /////
template <typename CoordiVecT, typename CoordiPortalT, typename CoordiT >
VTKM_EXEC_CONT vtkm::Id NNSVerify(CoordiVecT qc, CoordiPortalT coordiPortal, CoordiT& dis)
{
    dis = std::numeric_limits<CoordiT>::max();
    vtkm::Id nnpIdx = 0;

    for (vtkm::Int32 i = 0; i < coordiPortal.GetNumberOfValues(); i++)
    {
        CoordiT _dis = vtkm::Magnitude( coordiPortal.Get( i ) - qc );

        if (_dis < dis)
        {
            dis = _dis;
            nnpIdx = i;
        }
    }

    return nnpIdx;
}

class NearestNeighborSearchBruteForceWorklet : public vtkm::worklet::WorkletMapField
{
public:
    using ControlSignature = void(FieldIn qcIn,
                                  WholeArrayIn treeCoordiIn,
                                  FieldOut nnIdOut,
                                  FieldOut nnDisOut);
    using ExecutionSignature = void(_1, _2, _3, _4);

    VTKM_CONT
    NearestNeighborSearchBruteForceWorklet() {}

    template <typename CoordiVecType, typename CoordiPortalType, typename IdType, typename CoordiType>
    VTKM_EXEC void operator()(const CoordiVecType& qc,
                              const CoordiPortalType& coordiPortal,
                              IdType& nnId,
                              CoordiType& nnDis) const
    {
        nnDis = std::numeric_limits<CoordiType>::max();
        nnId = NNSVerify(qc, coordiPortal, nnDis);
    }
};

void TestKdTreeBuildNNS( vtkm::cont::DeviceAdapterId deviceId )
{
    vtkm::Int32 nTrainingPoints = 300000;
    vtkm::Int32 nTestingPoint   = 300000;

    std::vector<vtkm::Vec< vtkm::Float32, N_DIMS > > coordi;

    ///// randomly generate training points/////
    std::default_random_engine dre;
    std::uniform_real_distribution<vtkm::Float32> dr(0.0f, 10000.0f);

    for (vtkm::Int32 i = 0; i < nTrainingPoints; i++)
    {
        vtkm::Vec< vtkm::Float32, N_DIMS > tp;
        for( int d = 0; d < N_DIMS; ++d )
        {
            tp[ d ] = dr( dre );
        }
        coordi.push_back( tp );
    }

    ///// preprare data to build  kd tree /////
    auto coordi_Handle = vtkm::cont::make_ArrayHandle(coordi);

    vtkm::worklet::KdTree< N_DIMS > kdtree;

    auto t1 = std::chrono::high_resolution_clock::now();

    kdtree.Build( coordi_Handle, deviceId );

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "building took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";

    //Nearest Neighbor worklet Testing
    /// randomly generate testing points /////
    std::vector< vtkm::Vec< vtkm::Float32, N_DIMS > > qcVec;
    for (vtkm::Int32 i = 0; i < nTestingPoint; i++)
    {
        vtkm::Vec< vtkm::Float32, N_DIMS > tp;
        for( int d = 0; d < N_DIMS; ++d )
        {
            tp[ d ] = dr( dre );
        }
        qcVec.push_back( tp );
    }

    ///// preprare testing data /////
    auto qc_Handle = vtkm::cont::make_ArrayHandle(qcVec);
    std::vector< vtkm::Float32 > distances( qcVec.size(), std::numeric_limits< float >::max() );

    vtkm::cont::ArrayHandle<vtkm::Id> nnId_Handle;
    auto nnDis_Handle = vtkm::cont::make_ArrayHandle( distances );

    t1 = std::chrono::high_resolution_clock::now();

    // checkDevice( deviceId );

    kdtree.Run(
        coordi_Handle, qc_Handle, nnId_Handle, nnDis_Handle, deviceId );

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "searching took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";

    vtkm::cont::ArrayHandle<vtkm::Id> bfnnId_Handle;
    
    std::vector< vtkm::Float32 > distancesBF( qcVec.size(), std::numeric_limits< float >::max() );
    auto bfnnDis_Handle = vtkm::cont::make_ArrayHandle( distancesBF );

    NearestNeighborSearchBruteForceWorklet nnsbf3dWorklet;

    vtkm::worklet::DispatcherMapField< NearestNeighborSearchBruteForceWorklet > nnsbfDispatcher(
        nnsbf3dWorklet);

    t1 = std::chrono::high_resolution_clock::now();
    
    nnsbfDispatcher.Invoke(
        qc_Handle, vtkm::cont::make_ArrayHandle(coordi), bfnnId_Handle, bfnnDis_Handle );
    
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "brute force took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";

    // ///// verfity search result /////
    bool passTest = true;
    for (vtkm::Int32 i = 0; i < nTestingPoint; i++)
    {
        vtkm::Id workletIdx   = nnId_Handle.GetPortalControl().Get(i);
        vtkm::Id bfworkletIdx = bfnnId_Handle.GetPortalControl().Get(i);

        auto p1 = coordi[ workletIdx   ];
        auto p2 = coordi[ bfworkletIdx ];
        auto p3 = qcVec[ i ];
   
        auto d1 = vtkm::Magnitude( p3 - p1 );
        auto d2 = vtkm::Magnitude( p3 - p2 );

        if ( workletIdx != bfworkletIdx && d1 != d2 )
        {
            std::cout << workletIdx << "!=" << bfworkletIdx << " at " << i << std::endl;
            std::cout << vtkm::Magnitude( p3 - p1 ) << " vs " << vtkm::Magnitude( p3 - p2 ) << std::endl;
            passTest = false;
        }
    }

    if( passTest )
    {
        std::cout << "Passed Test\n";
    }

    VTKM_TEST_ASSERT( passTest, "Kd tree NN search result incorrect.");
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    vtkm::cont::testing::Testing::RunOnDevice( TestKdTreeBuildNNS, argc, argv );
}
