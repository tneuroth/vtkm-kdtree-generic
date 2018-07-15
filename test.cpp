#include <vtkm/worklet/DispatcherMapField.h>
#include "KDTree/KdTree.h"

#include <iostream>

using namespace std;

#define N_DIMS 10

namespace
{

    template< typename V_TP >
    static V_TP dist( const vtkm::Vec< V_TP, N_DIMS > & p1, const vtkm::Vec< V_TP, N_DIMS > & p2 )
    {
        V_TP sm = 0.0;
        for( int i = 0; i < N_DIMS; ++i )
        {
          V_TP diff = p1[ i ] - p2[ i ]; 
          sm += diff * diff;
        }
        return vtkm::Sqrt(  sm );
    }

using Algorithm = vtkm::cont::DeviceAdapterAlgorithm<VTKM_DEFAULT_DEVICE_ADAPTER_TAG>;

////brute force method /////
template <typename CoordiVecT, typename CoordiPortalT, typename CoordiT >
VTKM_EXEC_CONT vtkm::Id NNSVerify(CoordiVecT qc, CoordiPortalT coordiPortal, CoordiT& dis)
{
  dis = std::numeric_limits<CoordiT>::max();
  vtkm::Id nnpIdx = 0;

  for (vtkm::Int32 i = 0; i < coordiPortal.GetNumberOfValues(); i++)
  {
    CoordiT _dis = dist( coordiPortal.Get( i ), qc );
    
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
  using ControlSignature = void(FieldIn<> qcIn,
                                WholeArrayIn<> treeCoordiIn,
                                FieldOut<> nnIdOut,
                                FieldOut<> nnDisOut);
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

void TestKdTreeBuildNNS()
{
  vtkm::Int32 nTrainingPoints = 1000;
  vtkm::Int32 nTestingPoint = 1000;

  std::vector<vtkm::Vec< vtkm::Float32, N_DIMS > > coordi;

  ///// randomly generate training points/////
  std::default_random_engine dre;
  std::uniform_real_distribution<vtkm::Float32> dr(0.0f, 10.0f);

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

  // Run data
  vtkm::worklet::KdTree< N_DIMS > kdtree;
  kdtree.Build(coordi_Handle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG());

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

  vtkm::cont::ArrayHandle<vtkm::Id> nnId_Handle;
  vtkm::cont::ArrayHandle<vtkm::Float32> nnDis_Handle;

  kdtree.Run(
    coordi_Handle, qc_Handle, nnId_Handle, nnDis_Handle, VTKM_DEFAULT_DEVICE_ADAPTER_TAG());

  vtkm::cont::ArrayHandle<vtkm::Id> bfnnId_Handle;
  vtkm::cont::ArrayHandle<vtkm::Float32> bfnnDis_Handle;
  NearestNeighborSearchBruteForceWorklet nnsbf3dWorklet;
  vtkm::worklet::DispatcherMapField<NearestNeighborSearchBruteForceWorklet> nnsbfDispatcher(
    nnsbf3dWorklet);
  nnsbfDispatcher.Invoke(
    qc_Handle, vtkm::cont::make_ArrayHandle(coordi), bfnnId_Handle, bfnnDis_Handle);

  ///// verfity search result /////
  bool passTest = true;
  for (vtkm::Int32 i = 0; i < nTestingPoint; i++)
  {
    vtkm::Id workletIdx   = nnId_Handle.GetPortalControl().Get(i);
    vtkm::Id bfworkletIdx = bfnnId_Handle.GetPortalControl().Get(i);

    if (workletIdx != bfworkletIdx)
    {
      passTest = false;
    }
  }

  VTKM_TEST_ASSERT(passTest, "Kd tree NN search result incorrect.");
}

} // anonymous namespace

int main(int argc, char* argv[])
{
  int res = vtkm::cont::testing::Testing::Run( TestKdTreeBuildNNS );
  std::cout << res << "\n";
}
