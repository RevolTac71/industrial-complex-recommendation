import abc
import geopandas as gpd

class BaseSpatialPipeline(abc.ABC):
    """
    공간 분석 및 추천 파이프라인의 추상 베이스 클래스.
    추후 QGIS 연동 및 공간 분석 알고리즘 확정 시 이를 상속받아 구현합니다.
    """

    @abc.abstractmethod
    def run_analysis(
        self, 
        candidate_gdf: gpd.GeoDataFrame, 
        infrastructure_gdf: gpd.GeoDataFrame, 
        weights: dict[str, float]
    ) -> gpd.GeoDataFrame:
        """
        공간 분석 파이프라인 실행 메서드.
        
        Args:
            candidate_gdf: 산업단지 후보지 정보가 포함된 GeoDataFrame
            infrastructure_gdf: 주변 인프라 정보가 포함된 GeoDataFrame
            weights: 정주환경, 지역여건, 산업혁신, 물류여건 가중치 딕셔너리
            
        Returns:
            분석 및 점수 산출 결과가 추가된 GeoDataFrame
        """
        pass
