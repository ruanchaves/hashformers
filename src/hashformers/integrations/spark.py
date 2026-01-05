"""PySpark integration for hashformers.

This module provides a PySpark ML Transformer that uses hashformers
for word segmentation in distributed data processing pipelines.

HASH-406: Create PySpark Transformer for Databricks/AWS EMR
"""

from typing import Any, Iterator, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import PySpark - it's an optional dependency
try:
    from pyspark import keyword_only
    from pyspark.ml import Transformer
    from pyspark.ml.param import Param, Params, TypeConverters
    from pyspark.ml.param.shared import HasInputCol, HasOutputCol
    from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
    from pyspark.sql import DataFrame
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType, ArrayType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    # Create stub classes for when PySpark isn't installed
    class Transformer:
        pass
    class HasInputCol:
        pass
    class HasOutputCol:
        pass
    class DefaultParamsReadable:
        pass
    class DefaultParamsWritable:
        pass
    Params = object
    

class SparkHashformer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable
):
    """PySpark ML Transformer for word segmentation with hashformers.
    
    This transformer can be used in PySpark ML Pipelines for distributed
    processing of hashtag/text segmentation on Databricks or AWS EMR.
    
    Example:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.ml import Pipeline
        >>> from hashformers.integrations.spark import SparkHashformer
        >>> 
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = spark.createDataFrame([
        ...     ("weneedanationalpark",),
        ...     ("machinelearning",)
        ... ], ["hashtag"])
        >>> 
        >>> segmenter = SparkHashformer(
        ...     inputCol="hashtag",
        ...     outputCol="segmented",
        ...     segmenterModel="gpt2"
        ... )
        >>> 
        >>> # Use in a Pipeline
        >>> pipeline = Pipeline(stages=[segmenter])
        >>> model = pipeline.fit(df)
        >>> result = model.transform(df)
        >>> result.show()
    
    Args:
        inputCol: Input column name containing text to segment.
        outputCol: Output column name for segmented text.
        segmenterModel: Name or path of the segmenter model. Default is "gpt2".
        segmenterType: Type of segmenter ("incremental", "masked", "seq2seq").
        rerankerModel: Optional reranker model name or path.
        rerankerType: Type of reranker model.
        device: Device to run models on ("cuda" or "cpu").
    """
    
    segmenterModel = Param(
        Params._dummy(),
        "segmenterModel",
        "Name or path of the segmenter model",
        typeConverter=TypeConverters.toString
    ) if PYSPARK_AVAILABLE else None
    
    segmenterType = Param(
        Params._dummy(),
        "segmenterType",
        "Type of segmenter model (incremental, masked, seq2seq)",
        typeConverter=TypeConverters.toString
    ) if PYSPARK_AVAILABLE else None
    
    rerankerModel = Param(
        Params._dummy(),
        "rerankerModel",
        "Optional reranker model name or path",
        typeConverter=TypeConverters.toString
    ) if PYSPARK_AVAILABLE else None
    
    rerankerType = Param(
        Params._dummy(),
        "rerankerType",
        "Type of reranker model",
        typeConverter=TypeConverters.toString
    ) if PYSPARK_AVAILABLE else None
    
    device = Param(
        Params._dummy(),
        "device",
        "Device to run models on (cuda or cpu)",
        typeConverter=TypeConverters.toString
    ) if PYSPARK_AVAILABLE else None

    if PYSPARK_AVAILABLE:
        @keyword_only
        def __init__(
            self,
            inputCol: str = "text",
            outputCol: str = "segmented",
            segmenterModel: str = "gpt2",
            segmenterType: str = "incremental",
            rerankerModel: Optional[str] = None,
            rerankerType: Optional[str] = None,
            device: str = "cuda"
        ):
            super().__init__()
            self._setDefault(
                segmenterModel="gpt2",
                segmenterType="incremental",
                rerankerModel=None,
                rerankerType=None,
                device="cuda"
            )
            kwargs = self._input_kwargs
            self.setParams(**kwargs)
    else:
        def __init__(self, **kwargs):
            logger.warning(
                "PySpark is not installed. Install with: pip install pyspark"
            )
    
    if PYSPARK_AVAILABLE:
        @keyword_only
        def setParams(
            self,
            inputCol: str = "text",
            outputCol: str = "segmented",
            segmenterModel: str = "gpt2",
            segmenterType: str = "incremental",
            rerankerModel: Optional[str] = None,
            rerankerType: Optional[str] = None,
            device: str = "cuda"
        ):
            kwargs = self._input_kwargs
            return self._set(**kwargs)
    
    def getSegmenterModel(self) -> str:
        return self.getOrDefault(self.segmenterModel)
    
    def setSegmenterModel(self, value: str):
        return self._set(segmenterModel=value)
    
    def getSegmenterType(self) -> str:
        return self.getOrDefault(self.segmenterType)
    
    def setSegmenterType(self, value: str):
        return self._set(segmenterType=value)
    
    def getRerankerModel(self) -> Optional[str]:
        return self.getOrDefault(self.rerankerModel)
    
    def setRerankerModel(self, value: str):
        return self._set(rerankerModel=value)
    
    def getRerankerType(self) -> Optional[str]:
        return self.getOrDefault(self.rerankerType)
    
    def setRerankerType(self, value: str):
        return self._set(rerankerType=value)
    
    def getDevice(self) -> str:
        return self.getOrDefault(self.device)
    
    def setDevice(self, value: str):
        return self._set(device=value)
    
    def _transform(self, dataset: "DataFrame") -> "DataFrame":
        """Transform the input DataFrame by segmenting text.
        
        Args:
            dataset: Input PySpark DataFrame.
            
        Returns:
            DataFrame with additional column containing segmented text.
        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark is not installed")
        
        # Capture parameters for use in UDF closure
        segmenter_model = self.getSegmenterModel()
        segmenter_type = self.getSegmenterType()
        reranker_model = self.getRerankerModel()
        reranker_type = self.getRerankerType()
        device = self.getDevice()
        
        # Create a broadcast variable for model initialization params
        # The actual model is initialized per-partition to avoid serialization issues
        def segment_partition(iterator: Iterator) -> Iterator:
            """Process a partition of data with a single segmenter instance."""
            from hashformers import TransformerWordSegmenter
            
            # Initialize segmenter once per partition
            segmenter = TransformerWordSegmenter(
                segmenter_model_name_or_path=segmenter_model,
                segmenter_model_type=segmenter_type,
                reranker_model_name_or_path=reranker_model,
                reranker_model_type=reranker_type
            )
            
            for row in iterator:
                text = row[self.getInputCol()]
                if text:
                    result = segmenter.segment([text])[0]
                else:
                    result = ""
                yield (*row, result)
        
        # Define UDF for simple transformations
        @udf(returnType=StringType())
        def segment_text(text: str) -> str:
            if not text:
                return ""
            from hashformers import TransformerWordSegmenter
            segmenter = TransformerWordSegmenter(
                segmenter_model_name_or_path=segmenter_model,
                segmenter_model_type=segmenter_type,
                reranker_model_name_or_path=reranker_model,
                reranker_model_type=reranker_type
            )
            return segmenter.segment([text])[0]
        
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        
        return dataset.withColumn(output_col, segment_text(col(input_col)))

