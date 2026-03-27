# Pyspark Support Coverage

Tracking exhaustive coverage of the `pyspark` python package API.


## Detailed API

| Object Name | Type | Signature |
|---|---|---|
| `Accumulator` | Class | `(aid: int, value: T, accum_param: AccumulatorParam[T])` |
| `AccumulatorParam` | Class | `(...)` |
| `BarrierTaskContext` | Class | `(...)` |
| `BarrierTaskInfo` | Class | `(address: str)` |
| `BasicProfiler` | Class | `(ctx: SparkContext)` |
| `Broadcast` | Class | `(sc: Optional[SparkContext], value: Optional[T], pickle_registry: Optional[BroadcastPickleRegistry], path: Optional[str], sock_file: Optional[BinaryIO])` |
| `CPickleSerializer` | Object | `` |
| `HiveContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jhiveContext: Optional[JavaObject])` |
| `InheritableThread` | Class | `(target: Callable, args: Any, session: Optional[SparkSession], kwargs: Any)` |
| `MarshalSerializer` | Class | `(...)` |
| `Profiler` | Class | `(ctx: SparkContext)` |
| `RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `RDDBarrier` | Class | `(rdd: RDD[T])` |
| `Row` | Class | `(...)` |
| `SQLContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jsqlContext: Optional[JavaObject])` |
| `SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `SparkFiles` | Class | `()` |
| `SparkJobInfo` | Class | `(...)` |
| `SparkStageInfo` | Class | `(...)` |
| `StatusTracker` | Class | `(jtracker: JavaObject)` |
| `StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `TaskContext` | Class | `(...)` |
| `accumulators.Accumulator` | Class | `(aid: int, value: T, accum_param: AccumulatorParam[T])` |
| `accumulators.AccumulatorParam` | Class | `(...)` |
| `accumulators.AccumulatorTCPServer` | Class | `(server_address: Tuple[str, int], RequestHandlerClass: Type[socketserver.BaseRequestHandler], auth_token: str)` |
| `accumulators.AccumulatorUnixServer` | Class | `(socket_path: str, RequestHandlerClass: Type[socketserver.BaseRequestHandler])` |
| `accumulators.AddingAccumulatorParam` | Class | `(zero_value: U)` |
| `accumulators.COMPLEX_ACCUMULATOR_PARAM` | Object | `` |
| `accumulators.CPickleSerializer` | Object | `` |
| `accumulators.FLOAT_ACCUMULATOR_PARAM` | Object | `` |
| `accumulators.INT_ACCUMULATOR_PARAM` | Object | `` |
| `accumulators.PySparkRuntimeError` | Class | `(...)` |
| `accumulators.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `accumulators.SpecialAccumulatorIds` | Class | `(...)` |
| `accumulators.SupportsIAdd` | Class | `(...)` |
| `accumulators.T` | Object | `` |
| `accumulators.U` | Object | `` |
| `accumulators.UpdateRequestHandler` | Class | `(...)` |
| `accumulators.globs` | Object | `` |
| `accumulators.pickleSer` | Object | `` |
| `accumulators.read_int` | Function | `(stream)` |
| `broadcast` | Object | `` |
| `cloudpickle.CellType` | Object | `` |
| `cloudpickle.ChainMap` | Object | `` |
| `cloudpickle.CloudPickler` | Object | `` |
| `cloudpickle.DEFAULT_PROTOCOL` | Object | `` |
| `cloudpickle.DELETE_GLOBAL` | Object | `` |
| `cloudpickle.EXTENDED_ARG` | Object | `` |
| `cloudpickle.Enum` | Object | `` |
| `cloudpickle.GLOBAL_OPS` | Object | `` |
| `cloudpickle.HAVE_ARGUMENT` | Object | `` |
| `cloudpickle.LOAD_GLOBAL` | Object | `` |
| `cloudpickle.OrderedDict` | Object | `` |
| `cloudpickle.PYPY` | Object | `` |
| `cloudpickle.Pickler` | Class | `(file, protocol, buffer_callback)` |
| `cloudpickle.STORE_GLOBAL` | Object | `` |
| `cloudpickle.abc` | Object | `` |
| `cloudpickle.builtin_code_type` | Object | `` |
| `cloudpickle.builtins` | Object | `` |
| `cloudpickle.cloudpickle.CloudPickler` | Object | `` |
| `cloudpickle.cloudpickle.DEFAULT_PROTOCOL` | Object | `` |
| `cloudpickle.cloudpickle.DELETE_GLOBAL` | Object | `` |
| `cloudpickle.cloudpickle.EXTENDED_ARG` | Object | `` |
| `cloudpickle.cloudpickle.GLOBAL_OPS` | Object | `` |
| `cloudpickle.cloudpickle.HAVE_ARGUMENT` | Object | `` |
| `cloudpickle.cloudpickle.LOAD_GLOBAL` | Object | `` |
| `cloudpickle.cloudpickle.PYPY` | Object | `` |
| `cloudpickle.cloudpickle.Pickler` | Class | `(file, protocol, buffer_callback)` |
| `cloudpickle.cloudpickle.STORE_GLOBAL` | Object | `` |
| `cloudpickle.cloudpickle.builtin_code_type` | Object | `` |
| `cloudpickle.cloudpickle.dump` | Function | `(obj, file, protocol, buffer_callback)` |
| `cloudpickle.cloudpickle.dumps` | Function | `(obj, protocol, buffer_callback)` |
| `cloudpickle.cloudpickle.dynamic_subimport` | Function | `(name, vars)` |
| `cloudpickle.cloudpickle.instance` | Function | `(cls)` |
| `cloudpickle.cloudpickle.is_tornado_coroutine` | Function | `(func)` |
| `cloudpickle.cloudpickle.list_registry_pickle_by_value` | Function | `()` |
| `cloudpickle.cloudpickle.register_pickle_by_value` | Function | `(module)` |
| `cloudpickle.cloudpickle.subimport` | Function | `(name)` |
| `cloudpickle.cloudpickle.unregister_pickle_by_value` | Function | `(module)` |
| `cloudpickle.cloudpickle_fast.cloudpickle` | Object | `` |
| `cloudpickle.copyreg` | Object | `` |
| `cloudpickle.dataclasses` | Object | `` |
| `cloudpickle.dis` | Object | `` |
| `cloudpickle.dump` | Function | `(obj, file, protocol, buffer_callback)` |
| `cloudpickle.dumps` | Function | `(obj, protocol, buffer_callback)` |
| `cloudpickle.dynamic_subimport` | Function | `(name, vars)` |
| `cloudpickle.instance` | Function | `(cls)` |
| `cloudpickle.io` | Object | `` |
| `cloudpickle.is_tornado_coroutine` | Function | `(func)` |
| `cloudpickle.itertools` | Object | `` |
| `cloudpickle.list_registry_pickle_by_value` | Function | `()` |
| `cloudpickle.logging` | Object | `` |
| `cloudpickle.opcode` | Object | `` |
| `cloudpickle.pickle` | Object | `` |
| `cloudpickle.platform` | Object | `` |
| `cloudpickle.register_pickle_by_value` | Function | `(module)` |
| `cloudpickle.struct` | Object | `` |
| `cloudpickle.subimport` | Function | `(name)` |
| `cloudpickle.sys` | Object | `` |
| `cloudpickle.threading` | Object | `` |
| `cloudpickle.types` | Object | `` |
| `cloudpickle.typing` | Object | `` |
| `cloudpickle.unregister_pickle_by_value` | Function | `(module)` |
| `cloudpickle.uuid` | Object | `` |
| `cloudpickle.warnings` | Object | `` |
| `cloudpickle.weakref` | Object | `` |
| `conf.PySparkRuntimeError` | Class | `(...)` |
| `conf.SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `conf.is_remote_only` | Function | `() -> bool` |
| `context` | Object | `` |
| `core.broadcast.Broadcast` | Class | `(sc: Optional[SparkContext], value: Optional[T], pickle_registry: Optional[BroadcastPickleRegistry], path: Optional[str], sock_file: Optional[BinaryIO])` |
| `core.broadcast.BroadcastPickleRegistry` | Class | `()` |
| `core.broadcast.ChunkedStream` | Class | `(wrapped, buffer_size)` |
| `core.broadcast.PySparkRuntimeError` | Class | `(...)` |
| `core.broadcast.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `core.broadcast.T` | Object | `` |
| `core.broadcast.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `core.broadcast.pickle_protocol` | Object | `` |
| `core.broadcast.print_exec` | Function | `(stream: TextIO) -> None` |
| `core.context.Accumulator` | Class | `(aid: int, value: T, accum_param: AccumulatorParam[T])` |
| `core.context.AccumulatorParam` | Class | `(...)` |
| `core.context.AutoBatchedSerializer` | Class | `(serializer, bestSize)` |
| `core.context.BasicProfiler` | Class | `(ctx: SparkContext)` |
| `core.context.BatchedSerializer` | Class | `(serializer, batchSize)` |
| `core.context.Broadcast` | Class | `(sc: Optional[SparkContext], value: Optional[T], pickle_registry: Optional[BroadcastPickleRegistry], path: Optional[str], sock_file: Optional[BinaryIO])` |
| `core.context.BroadcastPickleRegistry` | Class | `()` |
| `core.context.CPickleSerializer` | Object | `` |
| `core.context.CallSite` | Object | `` |
| `core.context.ChunkedStream` | Class | `(wrapped, buffer_size)` |
| `core.context.DEFAULT_CONFIGS` | Object | `` |
| `core.context.DataType` | Class | `(...)` |
| `core.context.MemoryProfiler` | Class | `(ctx: SparkContext)` |
| `core.context.NoOpSerializer` | Class | `(...)` |
| `core.context.PairDeserializer` | Class | `(key_ser, val_ser)` |
| `core.context.ProfilerCollector` | Class | `(profiler_cls: Type[Profiler], udf_profiler_cls: Type[Profiler], memory_profiler_cls: Type[Profiler], dump_path: Optional[str])` |
| `core.context.PySparkRuntimeError` | Class | `(...)` |
| `core.context.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `core.context.ResourceInformation` | Class | `(name: str, addresses: List[str])` |
| `core.context.Serializer` | Class | `(...)` |
| `core.context.SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `core.context.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `core.context.SparkFiles` | Class | `()` |
| `core.context.StatusTracker` | Class | `(jtracker: JavaObject)` |
| `core.context.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `core.context.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `core.context.T` | Object | `` |
| `core.context.TaskContext` | Class | `(...)` |
| `core.context.U` | Object | `` |
| `core.context.UDFBasicProfiler` | Class | `(...)` |
| `core.context.UTF8Deserializer` | Class | `(use_unicode)` |
| `core.context.accumulators` | Object | `` |
| `core.context.first_spark_call` | Function | `()` |
| `core.context.launch_gateway` | Function | `(conf, popen_kwargs)` |
| `core.context.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `core.files.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `core.files.SparkFiles` | Class | `()` |
| `core.rdd.Aggregator` | Class | `(createCombiner, mergeValue, mergeCombiners)` |
| `core.rdd.AtomicType` | Class | `(...)` |
| `core.rdd.AtomicValue` | Object | `` |
| `core.rdd.AutoBatchedSerializer` | Class | `(serializer, bestSize)` |
| `core.rdd.BatchedSerializer` | Class | `(serializer, batchSize)` |
| `core.rdd.BoundedFloat` | Class | `(...)` |
| `core.rdd.CPickleSerializer` | Object | `` |
| `core.rdd.CartesianDeserializer` | Class | `(key_ser, val_ser)` |
| `core.rdd.CloudPickleSerializer` | Class | `(...)` |
| `core.rdd.DataFrame` | Class | `(...)` |
| `core.rdd.ExecutorResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, ExecutorResourceRequest]])` |
| `core.rdd.ExternalGroupBy` | Class | `(...)` |
| `core.rdd.ExternalMerger` | Class | `(aggregator, memory_limit, serializer, localdirs, scale, partitions, batch)` |
| `core.rdd.ExternalSorter` | Class | `(memory_limit, serializer)` |
| `core.rdd.K` | Object | `` |
| `core.rdd.NoOpSerializer` | Class | `(...)` |
| `core.rdd.NumberOrArray` | Object | `` |
| `core.rdd.PairDeserializer` | Class | `(key_ser, val_ser)` |
| `core.rdd.Partitioner` | Class | `(numPartitions: int, partitionFunc: Callable[[Any], int])` |
| `core.rdd.PipelinedRDD` | Class | `(prev: RDD[T], func: Callable[[int, Iterable[T]], Iterable[U]], preservesPartitioning: bool, isFromBarrier: bool)` |
| `core.rdd.PySparkRuntimeError` | Class | `(...)` |
| `core.rdd.PythonEvalType` | Class | `(...)` |
| `core.rdd.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `core.rdd.RDDBarrier` | Class | `(rdd: RDD[T])` |
| `core.rdd.RDDRangeSampler` | Class | `(lowerBound, upperBound, seed)` |
| `core.rdd.RDDSampler` | Class | `(withReplacement, fraction, seed)` |
| `core.rdd.RDDStratifiedSampler` | Class | `(withReplacement, fractions, seed)` |
| `core.rdd.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `core.rdd.ResultIterable` | Class | `(data: SizedIterable[T])` |
| `core.rdd.RowLike` | Object | `` |
| `core.rdd.S` | Object | `` |
| `core.rdd.SCCallSiteSync` | Class | `(sc)` |
| `core.rdd.Serializer` | Class | `(...)` |
| `core.rdd.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `core.rdd.StatCounter` | Class | `(values: Optional[Iterable[float]])` |
| `core.rdd.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `core.rdd.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `core.rdd.T` | Object | `` |
| `core.rdd.T_co` | Object | `` |
| `core.rdd.TaskResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, TaskResourceRequest]])` |
| `core.rdd.U` | Object | `` |
| `core.rdd.V` | Object | `` |
| `core.rdd.V1` | Object | `` |
| `core.rdd.V2` | Object | `` |
| `core.rdd.V3` | Object | `` |
| `core.rdd.fail_on_stopiteration` | Function | `(f: Callable) -> Callable` |
| `core.rdd.get_used_memory` | Function | `()` |
| `core.rdd.pack_long` | Function | `(value)` |
| `core.rdd.portable_hash` | Function | `(x: Hashable) -> int` |
| `core.rdd.python_cogroup` | Function | `(rdds, numPartitions)` |
| `core.rdd.python_full_outer_join` | Function | `(rdd, other, numPartitions)` |
| `core.rdd.python_join` | Function | `(rdd, other, numPartitions)` |
| `core.rdd.python_left_outer_join` | Function | `(rdd, other, numPartitions)` |
| `core.rdd.python_right_outer_join` | Function | `(rdd, other, numPartitions)` |
| `core.status.SparkJobInfo` | Class | `(...)` |
| `core.status.SparkStageInfo` | Class | `(...)` |
| `core.status.StatusTracker` | Class | `(jtracker: JavaObject)` |
| `daemon.UTF8Deserializer` | Class | `(use_unicode)` |
| `daemon.compute_real_exit_code` | Function | `(exit_code)` |
| `daemon.manager` | Function | `()` |
| `daemon.read_int` | Function | `(stream)` |
| `daemon.worker` | Function | `(sock, authenticated)` |
| `daemon.worker_main` | Function | `(infile, outfile)` |
| `daemon.worker_module` | Object | `` |
| `daemon.write_int` | Function | `(value, stream)` |
| `daemon.write_with_length` | Function | `(obj, stream)` |
| `errors.AnalysisException` | Class | `(...)` |
| `errors.ArithmeticException` | Class | `(...)` |
| `errors.ArrayIndexOutOfBoundsException` | Class | `(...)` |
| `errors.DateTimeException` | Class | `(...)` |
| `errors.IllegalArgumentException` | Class | `(...)` |
| `errors.NumberFormatException` | Class | `(...)` |
| `errors.ParseException` | Class | `(...)` |
| `errors.PickleException` | Class | `(...)` |
| `errors.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `errors.PySparkAttributeError` | Class | `(...)` |
| `errors.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `errors.PySparkImportError` | Class | `(...)` |
| `errors.PySparkIndexError` | Class | `(...)` |
| `errors.PySparkKeyError` | Class | `(...)` |
| `errors.PySparkNotImplementedError` | Class | `(...)` |
| `errors.PySparkPicklingError` | Class | `(...)` |
| `errors.PySparkRuntimeError` | Class | `(...)` |
| `errors.PySparkTypeError` | Class | `(...)` |
| `errors.PySparkValueError` | Class | `(...)` |
| `errors.PythonException` | Class | `(...)` |
| `errors.QueryContext` | Class | `(...)` |
| `errors.QueryContextType` | Class | `(...)` |
| `errors.QueryExecutionException` | Class | `(...)` |
| `errors.SessionNotSameException` | Class | `(...)` |
| `errors.SparkNoSuchElementException` | Class | `(...)` |
| `errors.SparkRuntimeException` | Class | `(...)` |
| `errors.SparkUpgradeException` | Class | `(...)` |
| `errors.StreamingPythonRunnerInitializationException` | Class | `(...)` |
| `errors.StreamingQueryException` | Class | `(...)` |
| `errors.TempTableAlreadyExistsException` | Class | `(...)` |
| `errors.UnknownException` | Class | `(...)` |
| `errors.UnsupportedOperationException` | Class | `(...)` |
| `errors.error_classes.ERROR_CLASSES_JSON` | Object | `` |
| `errors.error_classes.ERROR_CLASSES_MAP` | Object | `` |
| `errors.exceptions.base.AnalysisException` | Class | `(...)` |
| `errors.exceptions.base.ArithmeticException` | Class | `(...)` |
| `errors.exceptions.base.ArrayIndexOutOfBoundsException` | Class | `(...)` |
| `errors.exceptions.base.DateTimeException` | Class | `(...)` |
| `errors.exceptions.base.ErrorClassesReader` | Class | `()` |
| `errors.exceptions.base.IllegalArgumentException` | Class | `(...)` |
| `errors.exceptions.base.NumberFormatException` | Class | `(...)` |
| `errors.exceptions.base.ParseException` | Class | `(...)` |
| `errors.exceptions.base.PickleException` | Class | `(...)` |
| `errors.exceptions.base.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `errors.exceptions.base.PySparkAttributeError` | Class | `(...)` |
| `errors.exceptions.base.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `errors.exceptions.base.PySparkImportError` | Class | `(...)` |
| `errors.exceptions.base.PySparkIndexError` | Class | `(...)` |
| `errors.exceptions.base.PySparkKeyError` | Class | `(...)` |
| `errors.exceptions.base.PySparkLogger` | Class | `(name: str)` |
| `errors.exceptions.base.PySparkNotImplementedError` | Class | `(...)` |
| `errors.exceptions.base.PySparkPicklingError` | Class | `(...)` |
| `errors.exceptions.base.PySparkRuntimeError` | Class | `(...)` |
| `errors.exceptions.base.PySparkTypeError` | Class | `(...)` |
| `errors.exceptions.base.PySparkValueError` | Class | `(...)` |
| `errors.exceptions.base.PythonException` | Class | `(...)` |
| `errors.exceptions.base.QueryContext` | Class | `(...)` |
| `errors.exceptions.base.QueryContextType` | Class | `(...)` |
| `errors.exceptions.base.QueryExecutionException` | Class | `(...)` |
| `errors.exceptions.base.Row` | Class | `(...)` |
| `errors.exceptions.base.SessionNotSameException` | Class | `(...)` |
| `errors.exceptions.base.SparkNoSuchElementException` | Class | `(...)` |
| `errors.exceptions.base.SparkRuntimeException` | Class | `(...)` |
| `errors.exceptions.base.SparkUpgradeException` | Class | `(...)` |
| `errors.exceptions.base.StreamingPythonRunnerInitializationException` | Class | `(...)` |
| `errors.exceptions.base.StreamingQueryException` | Class | `(...)` |
| `errors.exceptions.base.T` | Object | `` |
| `errors.exceptions.base.TempTableAlreadyExistsException` | Class | `(...)` |
| `errors.exceptions.base.Traceback` | Class | `(tb: TracebackType, get_locals: Any)` |
| `errors.exceptions.base.UnknownException` | Class | `(...)` |
| `errors.exceptions.base.UnsupportedOperationException` | Class | `(...)` |
| `errors.exceptions.base.recover_python_exception` | Function | `(e: T) -> T` |
| `errors.exceptions.captured.AnalysisException` | Class | `(...)` |
| `errors.exceptions.captured.ArithmeticException` | Class | `(...)` |
| `errors.exceptions.captured.ArrayIndexOutOfBoundsException` | Class | `(...)` |
| `errors.exceptions.captured.BaseAnalysisException` | Class | `(...)` |
| `errors.exceptions.captured.BaseArithmeticException` | Class | `(...)` |
| `errors.exceptions.captured.BaseArrayIndexOutOfBoundsException` | Class | `(...)` |
| `errors.exceptions.captured.BaseDateTimeException` | Class | `(...)` |
| `errors.exceptions.captured.BaseIllegalArgumentException` | Class | `(...)` |
| `errors.exceptions.captured.BaseNoSuchElementException` | Class | `(...)` |
| `errors.exceptions.captured.BaseNumberFormatException` | Class | `(...)` |
| `errors.exceptions.captured.BaseParseException` | Class | `(...)` |
| `errors.exceptions.captured.BasePythonException` | Class | `(...)` |
| `errors.exceptions.captured.BaseQueryContext` | Class | `(...)` |
| `errors.exceptions.captured.BaseQueryExecutionException` | Class | `(...)` |
| `errors.exceptions.captured.BaseSparkRuntimeException` | Class | `(...)` |
| `errors.exceptions.captured.BaseSparkUpgradeException` | Class | `(...)` |
| `errors.exceptions.captured.BaseStreamingQueryException` | Class | `(...)` |
| `errors.exceptions.captured.BaseUnknownException` | Class | `(...)` |
| `errors.exceptions.captured.BaseUnsupportedOperationException` | Class | `(...)` |
| `errors.exceptions.captured.CapturedException` | Class | `(desc: Optional[str], stackTrace: Optional[str], cause: Optional[Py4JJavaError], origin: Optional[Py4JJavaError])` |
| `errors.exceptions.captured.DataFrameQueryContext` | Class | `(q: JavaObject)` |
| `errors.exceptions.captured.DateTimeException` | Class | `(...)` |
| `errors.exceptions.captured.IllegalArgumentException` | Class | `(...)` |
| `errors.exceptions.captured.NumberFormatException` | Class | `(...)` |
| `errors.exceptions.captured.ParseException` | Class | `(...)` |
| `errors.exceptions.captured.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `errors.exceptions.captured.PythonException` | Class | `(...)` |
| `errors.exceptions.captured.QueryContextType` | Class | `(...)` |
| `errors.exceptions.captured.QueryExecutionException` | Class | `(...)` |
| `errors.exceptions.captured.SQLQueryContext` | Class | `(q: JavaObject)` |
| `errors.exceptions.captured.SparkNoSuchElementException` | Class | `(...)` |
| `errors.exceptions.captured.SparkRuntimeException` | Class | `(...)` |
| `errors.exceptions.captured.SparkUpgradeException` | Class | `(...)` |
| `errors.exceptions.captured.StreamingQueryException` | Class | `(...)` |
| `errors.exceptions.captured.UnknownException` | Class | `(...)` |
| `errors.exceptions.captured.UnsupportedOperationException` | Class | `(...)` |
| `errors.exceptions.captured.capture_sql_exception` | Function | `(f: Callable[..., Any]) -> Callable[..., Any]` |
| `errors.exceptions.captured.convert_exception` | Function | `(e: Py4JJavaError) -> CapturedException` |
| `errors.exceptions.captured.install_exception_handler` | Function | `() -> None` |
| `errors.exceptions.captured.recover_python_exception` | Function | `(e: T) -> T` |
| `errors.exceptions.captured.unwrap_spark_exception` | Function | `() -> Iterator[Any]` |
| `errors.exceptions.connect.AnalysisException` | Class | `(...)` |
| `errors.exceptions.connect.ArithmeticException` | Class | `(...)` |
| `errors.exceptions.connect.ArrayIndexOutOfBoundsException` | Class | `(...)` |
| `errors.exceptions.connect.BaseAnalysisException` | Class | `(...)` |
| `errors.exceptions.connect.BaseArithmeticException` | Class | `(...)` |
| `errors.exceptions.connect.BaseArrayIndexOutOfBoundsException` | Class | `(...)` |
| `errors.exceptions.connect.BaseDateTimeException` | Class | `(...)` |
| `errors.exceptions.connect.BaseIllegalArgumentException` | Class | `(...)` |
| `errors.exceptions.connect.BaseNoSuchElementException` | Class | `(...)` |
| `errors.exceptions.connect.BaseNumberFormatException` | Class | `(...)` |
| `errors.exceptions.connect.BaseParseException` | Class | `(...)` |
| `errors.exceptions.connect.BasePickleException` | Class | `(...)` |
| `errors.exceptions.connect.BasePythonException` | Class | `(...)` |
| `errors.exceptions.connect.BaseQueryContext` | Class | `(...)` |
| `errors.exceptions.connect.BaseQueryExecutionException` | Class | `(...)` |
| `errors.exceptions.connect.BaseSparkRuntimeException` | Class | `(...)` |
| `errors.exceptions.connect.BaseSparkUpgradeException` | Class | `(...)` |
| `errors.exceptions.connect.BaseStreamingPythonRunnerInitException` | Class | `(...)` |
| `errors.exceptions.connect.BaseStreamingQueryException` | Class | `(...)` |
| `errors.exceptions.connect.BaseUnknownException` | Class | `(...)` |
| `errors.exceptions.connect.BaseUnsupportedOperationException` | Class | `(...)` |
| `errors.exceptions.connect.DataFrameQueryContext` | Class | `(q: pb2.FetchErrorDetailsResponse.QueryContext)` |
| `errors.exceptions.connect.DateTimeException` | Class | `(...)` |
| `errors.exceptions.connect.EXCEPTION_CLASS_MAPPING` | Object | `` |
| `errors.exceptions.connect.IllegalArgumentException` | Class | `(...)` |
| `errors.exceptions.connect.InvalidPlanInput` | Class | `(...)` |
| `errors.exceptions.connect.NumberFormatException` | Class | `(...)` |
| `errors.exceptions.connect.ParseException` | Class | `(...)` |
| `errors.exceptions.connect.PickleException` | Class | `(...)` |
| `errors.exceptions.connect.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `errors.exceptions.connect.PythonException` | Class | `(...)` |
| `errors.exceptions.connect.QueryContextType` | Class | `(...)` |
| `errors.exceptions.connect.QueryExecutionException` | Class | `(...)` |
| `errors.exceptions.connect.SQLQueryContext` | Class | `(q: pb2.FetchErrorDetailsResponse.QueryContext)` |
| `errors.exceptions.connect.SparkConnectException` | Class | `(...)` |
| `errors.exceptions.connect.SparkConnectGrpcException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], reason: Optional[str], sql_state: Optional[str], server_stacktrace: Optional[str], display_server_stacktrace: bool, contexts: Optional[List[BaseQueryContext]], grpc_status_code: grpc.StatusCode, breaking_change_info: Optional[Dict[str, Any]])` |
| `errors.exceptions.connect.SparkException` | Class | `(...)` |
| `errors.exceptions.connect.SparkNoSuchElementException` | Class | `(...)` |
| `errors.exceptions.connect.SparkRuntimeException` | Class | `(...)` |
| `errors.exceptions.connect.SparkUpgradeException` | Class | `(...)` |
| `errors.exceptions.connect.StreamingPythonRunnerInitializationException` | Class | `(...)` |
| `errors.exceptions.connect.StreamingQueryException` | Class | `(...)` |
| `errors.exceptions.connect.THIRD_PARTY_EXCEPTION_CLASS_MAPPING` | Object | `` |
| `errors.exceptions.connect.UnknownException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], reason: Optional[str], sql_state: Optional[str], server_stacktrace: Optional[str], display_server_stacktrace: bool, contexts: Optional[List[BaseQueryContext]], grpc_status_code: grpc.StatusCode, breaking_change_info: Optional[Dict[str, Any]])` |
| `errors.exceptions.connect.UnsupportedOperationException` | Class | `(...)` |
| `errors.exceptions.connect.convert_exception` | Function | `(info: ErrorInfo, truncated_message: str, resp: Optional[pb2.FetchErrorDetailsResponse], display_server_stacktrace: bool, grpc_status_code: grpc.StatusCode) -> SparkConnectException` |
| `errors.exceptions.connect.pb2` | Object | `` |
| `errors.exceptions.connect.recover_python_exception` | Function | `(e: T) -> T` |
| `errors.exceptions.tblib.Code` | Class | `(code: CodeType)` |
| `errors.exceptions.tblib.FRAME_RE` | Object | `` |
| `errors.exceptions.tblib.Frame` | Class | `(frame: FrameType, get_locals: Any)` |
| `errors.exceptions.tblib.LineCacheEntry` | Class | `(...)` |
| `errors.exceptions.tblib.Traceback` | Class | `(tb: TracebackType, get_locals: Any)` |
| `errors.exceptions.tblib.TracebackParseError` | Class | `(...)` |
| `errors.exceptions.tblib.get_all_locals` | Function | `(frame: FrameType) -> dict` |
| `errors.utils.ERROR_CLASSES_MAP` | Object | `` |
| `errors.utils.ErrorClassesReader` | Class | `()` |
| `errors.utils.FuncT` | Object | `` |
| `errors.utils.T` | Object | `` |
| `errors.utils.current_origin` | Function | `() -> threading.local` |
| `errors.utils.is_debugging_enabled` | Function | `() -> bool` |
| `errors.utils.set_current_origin` | Function | `(fragment: Optional[str], call_site: Optional[str]) -> None` |
| `errors.utils.with_origin_to_class` | Function | `(cls_or_ignores: Optional[Union[Type[T], List[str]]], ignores: Optional[List[str]]) -> Union[Type[T], Callable[[Type[T]], Type[T]]]` |
| `errors_doc_gen.ERROR_CLASSES_MAP` | Object | `` |
| `errors_doc_gen.generate_errors_doc` | Function | `(output_rst_file_path: str) -> None` |
| `files` | Object | `` |
| `inheritable_thread_target` | Function | `(f: Optional[Union[Callable, SparkSession]]) -> Callable` |
| `install.DEFAULT_HADOOP` | Object | `` |
| `install.DEFAULT_HIVE` | Object | `` |
| `install.SUPPORTED_HADOOP_VERSIONS` | Object | `` |
| `install.SUPPORTED_HIVE_VERSIONS` | Object | `` |
| `install.UNSUPPORTED_COMBINATIONS` | Object | `` |
| `install.checked_package_name` | Function | `(spark_version, hadoop_version, hive_version)` |
| `install.checked_versions` | Function | `(spark_version, hadoop_version, hive_version)` |
| `install.convert_old_hadoop_version` | Function | `(spark_version, hadoop_version)` |
| `install.download_to_file` | Function | `(response, path, chunk_size)` |
| `install.get_preferred_mirrors` | Function | `()` |
| `install.install_spark` | Function | `(dest, spark_version, hadoop_version, hive_version)` |
| `is_remote_only` | Function | `() -> bool` |
| `java_gateway.PySparkRuntimeError` | Class | `(...)` |
| `java_gateway.UTF8Deserializer` | Class | `(use_unicode)` |
| `java_gateway.ensure_callback_server_started` | Function | `(gw)` |
| `java_gateway.launch_gateway` | Function | `(conf, popen_kwargs)` |
| `java_gateway.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `java_gateway.read_int` | Function | `(stream)` |
| `join.ResultIterable` | Class | `(data: SizedIterable[T])` |
| `join.python_cogroup` | Function | `(rdds, numPartitions)` |
| `join.python_full_outer_join` | Function | `(rdd, other, numPartitions)` |
| `join.python_join` | Function | `(rdd, other, numPartitions)` |
| `join.python_left_outer_join` | Function | `(rdd, other, numPartitions)` |
| `join.python_right_outer_join` | Function | `(rdd, other, numPartitions)` |
| `keyword_only` | Function | `(func: _F) -> _F` |
| `logger.PySparkLogger` | Class | `(name: str)` |
| `logger.SPARK_LOG_SCHEMA` | Object | `` |
| `logger.logger.JSONFormatter` | Class | `(ensure_ascii: bool)` |
| `logger.logger.PySparkLogger` | Class | `(name: str)` |
| `logger.logger.SPARK_LOG_SCHEMA` | Object | `` |
| `logger.worker_io.DelegatingTextIOWrapper` | Class | `(delegate: TextIO)` |
| `logger.worker_io.JSONFormatter` | Class | `(ensure_ascii: bool)` |
| `logger.worker_io.JSONFormatterWithMarker` | Class | `(marker: str, worker_id: str, context_provider: Callable[[], dict[str, str]])` |
| `logger.worker_io.JsonOutput` | Class | `(delegate: TextIO, json_out: TextIO, logger_name: str, log_level: int, marker: str, worker_id: str, context_provider: Callable[[], dict[str, str]])` |
| `logger.worker_io.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `logger.worker_io.context_provider` | Function | `() -> dict[str, str]` |
| `loose_version.LooseVersion` | Class | `(vstring: Optional[str])` |
| `ml.Estimator` | Class | `(...)` |
| `ml.Model` | Class | `(...)` |
| `ml.Pipeline` | Class | `(stages: Optional[List[PipelineStage]])` |
| `ml.PipelineModel` | Class | `(stages: List[Transformer])` |
| `ml.PredictionModel` | Class | `(...)` |
| `ml.Predictor` | Class | `(...)` |
| `ml.TorchDistributor` | Class | `(num_processes: int, local_mode: bool, use_gpu: bool, _ssl_conf: str)` |
| `ml.Transformer` | Class | `(...)` |
| `ml.UnaryTransformer` | Class | `(...)` |
| `ml.base.DataFrame` | Class | `(...)` |
| `ml.base.DataType` | Class | `(...)` |
| `ml.base.Estimator` | Class | `(...)` |
| `ml.base.HasFeaturesCol` | Class | `()` |
| `ml.base.HasInputCol` | Class | `()` |
| `ml.base.HasLabelCol` | Class | `()` |
| `ml.base.HasOutputCol` | Class | `()` |
| `ml.base.HasPredictionCol` | Class | `()` |
| `ml.base.M` | Object | `` |
| `ml.base.Model` | Class | `(...)` |
| `ml.base.P` | Object | `` |
| `ml.base.ParamMap` | Object | `` |
| `ml.base.Params` | Class | `()` |
| `ml.base.PredictionModel` | Class | `(...)` |
| `ml.base.Predictor` | Class | `(...)` |
| `ml.base.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `ml.base.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `ml.base.T` | Object | `` |
| `ml.base.Transformer` | Class | `(...)` |
| `ml.base.UnaryTransformer` | Class | `(...)` |
| `ml.base.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.base.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.base.udf` | Function | `(f: Optional[Union[Callable[..., Any], DataTypeOrString]], returnType: DataTypeOrString, useArrow: Optional[bool]) -> Union[UserDefinedFunctionLike, Callable[[Callable[..., Any]], UserDefinedFunctionLike]]` |
| `ml.classification.BinaryLogisticRegressionSummary` | Class | `(...)` |
| `ml.classification.BinaryLogisticRegressionTrainingSummary` | Class | `(...)` |
| `ml.classification.BinaryRandomForestClassificationSummary` | Class | `(...)` |
| `ml.classification.BinaryRandomForestClassificationTrainingSummary` | Class | `(...)` |
| `ml.classification.CM` | Object | `` |
| `ml.classification.ClassificationModel` | Class | `(...)` |
| `ml.classification.Classifier` | Class | `(...)` |
| `ml.classification.DataFrame` | Class | `(...)` |
| `ml.classification.DecisionTreeClassificationModel` | Class | `(...)` |
| `ml.classification.DecisionTreeClassifier` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, probabilityCol: str, rawPredictionCol: str, maxDepth: int, maxBins: int, minInstancesPerNode: int, minInfoGain: float, maxMemoryInMB: int, cacheNodeIds: bool, checkpointInterval: int, impurity: str, seed: Optional[int], weightCol: Optional[str], leafCol: str, minWeightFractionPerNode: float)` |
| `ml.classification.DecisionTreeRegressionModel` | Class | `(...)` |
| `ml.classification.DefaultParamsReader` | Class | `(cls: Type[DefaultParamsReadable[RL]])` |
| `ml.classification.DefaultParamsWriter` | Class | `(instance: Params)` |
| `ml.classification.Estimator` | Class | `(...)` |
| `ml.classification.F` | Object | `` |
| `ml.classification.FMClassificationModel` | Class | `(...)` |
| `ml.classification.FMClassificationSummary` | Class | `(...)` |
| `ml.classification.FMClassificationTrainingSummary` | Class | `(...)` |
| `ml.classification.FMClassifier` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, probabilityCol: str, rawPredictionCol: str, factorSize: int, fitIntercept: bool, fitLinear: bool, regParam: float, miniBatchFraction: float, initStd: float, maxIter: int, stepSize: float, tol: float, solver: str, thresholds: Optional[List[float]], seed: Optional[int])` |
| `ml.classification.GBTClassificationModel` | Class | `(...)` |
| `ml.classification.GBTClassifier` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxDepth: int, maxBins: int, minInstancesPerNode: int, minInfoGain: float, maxMemoryInMB: int, cacheNodeIds: bool, checkpointInterval: int, lossType: str, maxIter: int, stepSize: float, seed: Optional[int], subsamplingRate: float, impurity: str, featureSubsetStrategy: str, validationTol: float, validationIndicatorCol: Optional[str], leafCol: str, minWeightFractionPerNode: float, weightCol: Optional[str])` |
| `ml.classification.HasAggregationDepth` | Class | `()` |
| `ml.classification.HasBlockSize` | Class | `()` |
| `ml.classification.HasElasticNetParam` | Class | `()` |
| `ml.classification.HasFitIntercept` | Class | `()` |
| `ml.classification.HasMaxBlockSizeInMB` | Class | `()` |
| `ml.classification.HasMaxIter` | Class | `()` |
| `ml.classification.HasParallelism` | Class | `()` |
| `ml.classification.HasProbabilityCol` | Class | `()` |
| `ml.classification.HasRawPredictionCol` | Class | `()` |
| `ml.classification.HasRegParam` | Class | `()` |
| `ml.classification.HasSeed` | Class | `()` |
| `ml.classification.HasSolver` | Class | `()` |
| `ml.classification.HasStandardization` | Class | `()` |
| `ml.classification.HasStepSize` | Class | `()` |
| `ml.classification.HasThreshold` | Class | `()` |
| `ml.classification.HasThresholds` | Class | `()` |
| `ml.classification.HasTol` | Class | `()` |
| `ml.classification.HasTrainingSummary` | Class | `(...)` |
| `ml.classification.HasWeightCol` | Class | `()` |
| `ml.classification.JPM` | Object | `` |
| `ml.classification.JavaMLReadable` | Class | `(...)` |
| `ml.classification.JavaMLWritable` | Class | `(...)` |
| `ml.classification.JavaMLWriter` | Class | `(instance: JavaMLWritable)` |
| `ml.classification.JavaParams` | Class | `(...)` |
| `ml.classification.JavaPredictionModel` | Class | `(...)` |
| `ml.classification.JavaPredictor` | Class | `(...)` |
| `ml.classification.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.classification.LinearSVC` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxIter: int, regParam: float, tol: float, rawPredictionCol: str, fitIntercept: bool, standardization: bool, threshold: float, weightCol: Optional[str], aggregationDepth: int, maxBlockSizeInMB: float)` |
| `ml.classification.LinearSVCModel` | Class | `(...)` |
| `ml.classification.LinearSVCSummary` | Class | `(...)` |
| `ml.classification.LinearSVCTrainingSummary` | Class | `(...)` |
| `ml.classification.LogisticRegression` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxIter: int, regParam: float, elasticNetParam: float, tol: float, fitIntercept: bool, threshold: float, thresholds: Optional[List[float]], probabilityCol: str, rawPredictionCol: str, standardization: bool, weightCol: Optional[str], aggregationDepth: int, family: str, lowerBoundsOnCoefficients: Optional[Matrix], upperBoundsOnCoefficients: Optional[Matrix], lowerBoundsOnIntercepts: Optional[Vector], upperBoundsOnIntercepts: Optional[Vector], maxBlockSizeInMB: float)` |
| `ml.classification.LogisticRegressionModel` | Class | `(...)` |
| `ml.classification.LogisticRegressionSummary` | Class | `(...)` |
| `ml.classification.LogisticRegressionTrainingSummary` | Class | `(...)` |
| `ml.classification.MF` | Object | `` |
| `ml.classification.MLReadable` | Class | `(...)` |
| `ml.classification.MLReader` | Class | `()` |
| `ml.classification.MLWritable` | Class | `(...)` |
| `ml.classification.MLWriter` | Class | `()` |
| `ml.classification.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `ml.classification.Model` | Class | `(...)` |
| `ml.classification.MultilayerPerceptronClassificationModel` | Class | `(...)` |
| `ml.classification.MultilayerPerceptronClassificationSummary` | Class | `(...)` |
| `ml.classification.MultilayerPerceptronClassificationTrainingSummary` | Class | `(...)` |
| `ml.classification.MultilayerPerceptronClassifier` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxIter: int, tol: float, seed: Optional[int], layers: Optional[List[int]], blockSize: int, stepSize: float, solver: str, initialWeights: Optional[Vector], probabilityCol: str, rawPredictionCol: str)` |
| `ml.classification.NaiveBayes` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, probabilityCol: str, rawPredictionCol: str, smoothing: float, modelType: str, thresholds: Optional[List[float]], weightCol: Optional[str])` |
| `ml.classification.NaiveBayesModel` | Class | `(...)` |
| `ml.classification.OneVsRest` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, rawPredictionCol: str, classifier: Optional[Classifier[CM]], weightCol: Optional[str], parallelism: int)` |
| `ml.classification.OneVsRestModel` | Class | `(models: List[ClassificationModel])` |
| `ml.classification.OneVsRestModelReader` | Class | `(cls: Type[OneVsRestModel])` |
| `ml.classification.OneVsRestModelWriter` | Class | `(instance: OneVsRestModel)` |
| `ml.classification.OneVsRestReader` | Class | `(cls: Type[OneVsRest])` |
| `ml.classification.OneVsRestWriter` | Class | `(instance: OneVsRest)` |
| `ml.classification.P` | Object | `` |
| `ml.classification.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.classification.ParamMap` | Object | `` |
| `ml.classification.Params` | Class | `()` |
| `ml.classification.PredictionModel` | Class | `(...)` |
| `ml.classification.Predictor` | Class | `(...)` |
| `ml.classification.ProbabilisticClassificationModel` | Class | `(...)` |
| `ml.classification.ProbabilisticClassifier` | Class | `(...)` |
| `ml.classification.RandomForestClassificationModel` | Class | `(...)` |
| `ml.classification.RandomForestClassificationSummary` | Class | `(...)` |
| `ml.classification.RandomForestClassificationTrainingSummary` | Class | `(...)` |
| `ml.classification.RandomForestClassifier` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, probabilityCol: str, rawPredictionCol: str, maxDepth: int, maxBins: int, minInstancesPerNode: int, minInfoGain: float, maxMemoryInMB: int, cacheNodeIds: bool, checkpointInterval: int, impurity: str, numTrees: int, featureSubsetStrategy: str, seed: Optional[int], subsamplingRate: float, leafCol: str, minWeightFractionPerNode: float, weightCol: Optional[str], bootstrap: Optional[bool])` |
| `ml.classification.Row` | Class | `(...)` |
| `ml.classification.SF` | Class | `(...)` |
| `ml.classification.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `ml.classification.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.classification.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `ml.classification.T` | Object | `` |
| `ml.classification.TypeConverters` | Class | `(...)` |
| `ml.classification.Vector` | Class | `(...)` |
| `ml.classification.globs` | Object | `` |
| `ml.classification.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.classification.inheritable_thread_target` | Function | `(f: Optional[Union[Callable, SparkSession]]) -> Callable` |
| `ml.classification.is_remote` | Function | `() -> bool` |
| `ml.classification.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.classification.sc` | Object | `` |
| `ml.classification.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.classification.spark` | Object | `` |
| `ml.classification.temp_path` | Object | `` |
| `ml.classification.try_remote_attribute_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.classification.try_remote_read` | Function | `(f: FuncT) -> FuncT` |
| `ml.classification.try_remote_write` | Function | `(f: FuncT) -> FuncT` |
| `ml.clustering.BisectingKMeans` | Class | `(featuresCol: str, predictionCol: str, maxIter: int, seed: Optional[int], k: int, minDivisibleClusterSize: float, distanceMeasure: str, weightCol: Optional[str])` |
| `ml.clustering.BisectingKMeansModel` | Class | `(...)` |
| `ml.clustering.BisectingKMeansSummary` | Class | `(...)` |
| `ml.clustering.ClusteringSummary` | Class | `(...)` |
| `ml.clustering.DataFrame` | Class | `(...)` |
| `ml.clustering.DistributedLDAModel` | Class | `(...)` |
| `ml.clustering.GaussianMixture` | Class | `(featuresCol: str, predictionCol: str, k: int, probabilityCol: str, tol: float, maxIter: int, seed: Optional[int], aggregationDepth: int, weightCol: Optional[str])` |
| `ml.clustering.GaussianMixtureModel` | Class | `(...)` |
| `ml.clustering.GaussianMixtureSummary` | Class | `(...)` |
| `ml.clustering.GeneralJavaMLWritable` | Class | `(...)` |
| `ml.clustering.HasAggregationDepth` | Class | `()` |
| `ml.clustering.HasCheckpointInterval` | Class | `()` |
| `ml.clustering.HasDistanceMeasure` | Class | `()` |
| `ml.clustering.HasFeaturesCol` | Class | `()` |
| `ml.clustering.HasMaxBlockSizeInMB` | Class | `()` |
| `ml.clustering.HasMaxIter` | Class | `()` |
| `ml.clustering.HasPredictionCol` | Class | `()` |
| `ml.clustering.HasProbabilityCol` | Class | `()` |
| `ml.clustering.HasSeed` | Class | `()` |
| `ml.clustering.HasSolver` | Class | `()` |
| `ml.clustering.HasTol` | Class | `()` |
| `ml.clustering.HasTrainingSummary` | Class | `(...)` |
| `ml.clustering.HasWeightCol` | Class | `()` |
| `ml.clustering.JavaEstimator` | Class | `(...)` |
| `ml.clustering.JavaMLReadable` | Class | `(...)` |
| `ml.clustering.JavaMLWritable` | Class | `(...)` |
| `ml.clustering.JavaModel` | Class | `(java_model: Optional[JavaObject])` |
| `ml.clustering.JavaParams` | Class | `(...)` |
| `ml.clustering.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.clustering.KMeans` | Class | `(featuresCol: str, predictionCol: str, k: int, initMode: str, initSteps: int, tol: float, maxIter: int, seed: Optional[int], distanceMeasure: str, weightCol: Optional[str], solver: str, maxBlockSizeInMB: float)` |
| `ml.clustering.KMeansModel` | Class | `(...)` |
| `ml.clustering.KMeansSummary` | Class | `(...)` |
| `ml.clustering.LDA` | Class | `(featuresCol: str, maxIter: int, seed: Optional[int], checkpointInterval: int, k: int, optimizer: str, learningOffset: float, learningDecay: float, subsamplingRate: float, optimizeDocConcentration: bool, docConcentration: Optional[List[float]], topicConcentration: Optional[float], topicDistributionCol: str, keepLastCheckpoint: bool)` |
| `ml.clustering.LDAModel` | Class | `(...)` |
| `ml.clustering.LocalLDAModel` | Class | `(...)` |
| `ml.clustering.M` | Object | `` |
| `ml.clustering.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `ml.clustering.MultivariateGaussian` | Class | `(mean: Vector, cov: Matrix)` |
| `ml.clustering.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.clustering.Params` | Class | `()` |
| `ml.clustering.PowerIterationClustering` | Class | `(k: int, maxIter: int, initMode: str, srcCol: str, dstCol: str, weightCol: Optional[str])` |
| `ml.clustering.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.clustering.TypeConverters` | Class | `(...)` |
| `ml.clustering.Vector` | Class | `(...)` |
| `ml.clustering.globs` | Object | `` |
| `ml.clustering.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.clustering.invoke_helper_relation` | Function | `(method: str, args: Any) -> ConnectDataFrame` |
| `ml.clustering.is_remote` | Function | `() -> bool` |
| `ml.clustering.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.clustering.sc` | Object | `` |
| `ml.clustering.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.clustering.spark` | Object | `` |
| `ml.clustering.temp_path` | Object | `` |
| `ml.clustering.try_remote_attribute_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.common.AutoBatchedSerializer` | Class | `(serializer, bestSize)` |
| `ml.common.C` | Object | `` |
| `ml.common.CPickleSerializer` | Object | `` |
| `ml.common.DataFrame` | Class | `(...)` |
| `ml.common.JavaObjectOrPickleDump` | Object | `` |
| `ml.common.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `ml.common.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `ml.common.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.common.callJavaFunc` | Function | `(sc: pyspark.core.context.SparkContext, func: Callable[..., JavaObjectOrPickleDump], args: Any) -> JavaObjectOrPickleDump` |
| `ml.common.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.common.is_remote_only` | Function | `() -> bool` |
| `ml.connect.Estimator` | Class | `(...)` |
| `ml.connect.Evaluator` | Class | `(...)` |
| `ml.connect.Model` | Class | `(...)` |
| `ml.connect.Pipeline` | Class | `(stages: Optional[List[Params]])` |
| `ml.connect.PipelineModel` | Class | `(stages: Optional[List[Params]])` |
| `ml.connect.Transformer` | Class | `(...)` |
| `ml.connect.base.DataFrame` | Class | `(...)` |
| `ml.connect.base.Estimator` | Class | `(...)` |
| `ml.connect.base.Evaluator` | Class | `(...)` |
| `ml.connect.base.HasFeaturesCol` | Class | `()` |
| `ml.connect.base.HasLabelCol` | Class | `()` |
| `ml.connect.base.HasPredictionCol` | Class | `()` |
| `ml.connect.base.M` | Object | `` |
| `ml.connect.base.Model` | Class | `(...)` |
| `ml.connect.base.ParamMap` | Object | `` |
| `ml.connect.base.Params` | Class | `()` |
| `ml.connect.base.PredictionModel` | Class | `(...)` |
| `ml.connect.base.Predictor` | Class | `(...)` |
| `ml.connect.base.Transformer` | Class | `(...)` |
| `ml.connect.base.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.connect.base.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.connect.classification.CoreModelReadWrite` | Class | `(...)` |
| `ml.connect.classification.DataFrame` | Class | `(...)` |
| `ml.connect.classification.HasBatchSize` | Class | `()` |
| `ml.connect.classification.HasFitIntercept` | Class | `()` |
| `ml.connect.classification.HasLearningRate` | Class | `()` |
| `ml.connect.classification.HasMaxIter` | Class | `()` |
| `ml.connect.classification.HasMomentum` | Class | `()` |
| `ml.connect.classification.HasNumTrainWorkers` | Class | `()` |
| `ml.connect.classification.HasProbabilityCol` | Class | `()` |
| `ml.connect.classification.HasSeed` | Class | `()` |
| `ml.connect.classification.HasTol` | Class | `()` |
| `ml.connect.classification.HasWeightCol` | Class | `()` |
| `ml.connect.classification.LogisticRegression` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, probabilityCol: str, maxIter: int, tol: float, numTrainWorkers: int, batchSize: int, learningRate: float, momentum: float, seed: int)` |
| `ml.connect.classification.LogisticRegressionModel` | Class | `(torch_model: Any, num_features: Optional[int], num_classes: Optional[int])` |
| `ml.connect.classification.ParamsReadWrite` | Class | `(...)` |
| `ml.connect.classification.PredictionModel` | Class | `(...)` |
| `ml.connect.classification.Predictor` | Class | `(...)` |
| `ml.connect.classification.TorchDistributor` | Class | `(num_processes: int, local_mode: bool, use_gpu: bool, _ssl_conf: str)` |
| `ml.connect.classification.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.connect.classification.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.connect.classification.sf` | Object | `` |
| `ml.connect.evaluation.BinaryClassificationEvaluator` | Class | `(metricName: str, labelCol: str, probabilityCol: str)` |
| `ml.connect.evaluation.DataFrame` | Class | `(...)` |
| `ml.connect.evaluation.Evaluator` | Class | `(...)` |
| `ml.connect.evaluation.HasLabelCol` | Class | `()` |
| `ml.connect.evaluation.HasPredictionCol` | Class | `()` |
| `ml.connect.evaluation.HasProbabilityCol` | Class | `()` |
| `ml.connect.evaluation.MulticlassClassificationEvaluator` | Class | `(metricName: str, labelCol: str, predictionCol: str)` |
| `ml.connect.evaluation.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.connect.evaluation.Params` | Class | `()` |
| `ml.connect.evaluation.ParamsReadWrite` | Class | `(...)` |
| `ml.connect.evaluation.RegressionEvaluator` | Class | `(metricName: str, labelCol: str, predictionCol: str)` |
| `ml.connect.evaluation.TypeConverters` | Class | `(...)` |
| `ml.connect.evaluation.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.connect.feature.ArrayAssembler` | Class | `(inputCols: Optional[List[str]], outputCol: Optional[str], featureSizes: Optional[List[int]], handleInvalid: Optional[str])` |
| `ml.connect.feature.CoreModelReadWrite` | Class | `(...)` |
| `ml.connect.feature.DataFrame` | Class | `(...)` |
| `ml.connect.feature.Estimator` | Class | `(...)` |
| `ml.connect.feature.HasFeatureSizes` | Class | `()` |
| `ml.connect.feature.HasHandleInvalid` | Class | `()` |
| `ml.connect.feature.HasInputCol` | Class | `()` |
| `ml.connect.feature.HasInputCols` | Class | `()` |
| `ml.connect.feature.HasOutputCol` | Class | `()` |
| `ml.connect.feature.MaxAbsScaler` | Class | `(inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.connect.feature.MaxAbsScalerModel` | Class | `(max_abs_values: Optional[np.ndarray], n_samples_seen: Optional[int])` |
| `ml.connect.feature.Model` | Class | `(...)` |
| `ml.connect.feature.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.connect.feature.Params` | Class | `()` |
| `ml.connect.feature.ParamsReadWrite` | Class | `(...)` |
| `ml.connect.feature.StandardScaler` | Class | `(inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.connect.feature.StandardScalerModel` | Class | `(mean_values: Optional[np.ndarray], std_values: Optional[np.ndarray], n_samples_seen: Optional[int])` |
| `ml.connect.feature.Transformer` | Class | `(...)` |
| `ml.connect.feature.TypeConverters` | Class | `(...)` |
| `ml.connect.feature.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.connect.functions.Column` | Class | `(...)` |
| `ml.connect.functions.PyMLFunctions` | Object | `` |
| `ml.connect.functions.UserDefinedFunctionLike` | Class | `(...)` |
| `ml.connect.functions.array_to_vector` | Function | `(col: Column) -> Column` |
| `ml.connect.functions.predict_batch_udf` | Function | `(args: Any, kwargs: Any) -> UserDefinedFunctionLike` |
| `ml.connect.functions.vector_to_array` | Function | `(col: Column, dtype: str) -> Column` |
| `ml.connect.io_utils.CoreModelReadWrite` | Class | `(...)` |
| `ml.connect.io_utils.MetaAlgorithmReadWrite` | Class | `(...)` |
| `ml.connect.io_utils.Params` | Class | `()` |
| `ml.connect.io_utils.ParamsReadWrite` | Class | `(...)` |
| `ml.connect.io_utils.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.connect.io_utils.is_remote` | Function | `() -> bool` |
| `ml.connect.io_utils.pyspark_version` | Object | `` |
| `ml.connect.pipeline.DataFrame` | Class | `(...)` |
| `ml.connect.pipeline.Estimator` | Class | `(...)` |
| `ml.connect.pipeline.MetaAlgorithmReadWrite` | Class | `(...)` |
| `ml.connect.pipeline.Model` | Class | `(...)` |
| `ml.connect.pipeline.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.connect.pipeline.ParamMap` | Object | `` |
| `ml.connect.pipeline.Params` | Class | `()` |
| `ml.connect.pipeline.ParamsReadWrite` | Class | `(...)` |
| `ml.connect.pipeline.Pipeline` | Class | `(stages: Optional[List[Params]])` |
| `ml.connect.pipeline.PipelineModel` | Class | `(stages: Optional[List[Params]])` |
| `ml.connect.pipeline.Transformer` | Class | `(...)` |
| `ml.connect.pipeline.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.connect.pipeline.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.connect.pipeline.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.connect.proto.AttributeRelation` | Class | `(ref_id: str, methods: List[pb2.Fetch.Method], child: Optional[LogicalPlan])` |
| `ml.connect.proto.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `ml.connect.proto.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `ml.connect.proto.TransformerRelation` | Class | `(child: Optional[LogicalPlan], name: str, ml_params: pb2.MlParams, uid: str, is_model: bool)` |
| `ml.connect.proto.check_dependencies` | Function | `(mod_name: str) -> None` |
| `ml.connect.proto.pb2` | Object | `` |
| `ml.connect.readwrite.JavaMLReadable` | Class | `(...)` |
| `ml.connect.readwrite.JavaMLWritable` | Class | `(...)` |
| `ml.connect.readwrite.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.connect.readwrite.MLReader` | Class | `()` |
| `ml.connect.readwrite.MLWriter` | Class | `()` |
| `ml.connect.readwrite.RL` | Object | `` |
| `ml.connect.readwrite.RemoteMLReader` | Class | `(clazz: Type[JavaMLReadable[RL]])` |
| `ml.connect.readwrite.RemoteMLWriter` | Class | `(instance: JavaMLWritable)` |
| `ml.connect.readwrite.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `ml.connect.readwrite.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `ml.connect.readwrite.check_dependencies` | Function | `(mod_name: str) -> None` |
| `ml.connect.readwrite.deserialize` | Function | `(ml_command_result_properties: Dict[str, Any]) -> Any` |
| `ml.connect.readwrite.deserialize_param` | Function | `(literal: pb2.Expression.Literal) -> Any` |
| `ml.connect.readwrite.pb2` | Object | `` |
| `ml.connect.readwrite.serialize_ml_params` | Function | `(instance: Params, client: SparkConnectClient) -> pb2.MlParams` |
| `ml.connect.serialize.DataType` | Class | `(...)` |
| `ml.connect.serialize.DenseMatrix` | Class | `(numRows: int, numCols: int, values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `ml.connect.serialize.DenseVector` | Class | `(ar: Union[bytes, np.ndarray, Iterable[float]])` |
| `ml.connect.serialize.Params` | Class | `()` |
| `ml.connect.serialize.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `ml.connect.serialize.SparseMatrix` | Class | `(numRows: int, numCols: int, colPtrs: Union[bytes, Iterable[int]], rowIndices: Union[bytes, Iterable[int]], values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `ml.connect.serialize.SparseVector` | Class | `(size: int, args: Union[bytes, Tuple[int, float], Iterable[float], Iterable[Tuple[int, float]], Dict[int, float]])` |
| `ml.connect.serialize.build_float_list` | Function | `(value: List[float]) -> pb2.Expression.Literal` |
| `ml.connect.serialize.build_int_list` | Function | `(value: List[int]) -> pb2.Expression.Literal` |
| `ml.connect.serialize.build_proto_udt` | Function | `(jvm_class: str) -> pb2.DataType` |
| `ml.connect.serialize.check_dependencies` | Function | `(mod_name: str) -> None` |
| `ml.connect.serialize.deserialize` | Function | `(ml_command_result_properties: Dict[str, Any]) -> Any` |
| `ml.connect.serialize.deserialize_param` | Function | `(literal: pb2.Expression.Literal) -> Any` |
| `ml.connect.serialize.literal_null` | Function | `() -> pb2.Expression.Literal` |
| `ml.connect.serialize.pb2` | Object | `` |
| `ml.connect.serialize.proto_matrix_udt` | Object | `` |
| `ml.connect.serialize.proto_vector_udt` | Object | `` |
| `ml.connect.serialize.serialize` | Function | `(client: SparkConnectClient, args: Any) -> List[Any]` |
| `ml.connect.serialize.serialize_ml_params` | Function | `(instance: Params, client: SparkConnectClient) -> pb2.MlParams` |
| `ml.connect.serialize.serialize_ml_params_values` | Function | `(values: Dict[str, Any], client: SparkConnectClient) -> pb2.MlParams` |
| `ml.connect.serialize.serialize_param` | Function | `(value: Any, client: SparkConnectClient) -> pb2.Expression.Literal` |
| `ml.connect.summarizer.DataFrame` | Class | `(...)` |
| `ml.connect.summarizer.SummarizerAggState` | Class | `(input_array: np.ndarray)` |
| `ml.connect.summarizer.aggregate_dataframe` | Function | `(dataframe: Union[DataFrame, pd.DataFrame], input_col_names: List[str], local_agg_fn: Callable[[pd.DataFrame], Any], merge_agg_state: Callable[[Any, Any], Any], agg_state_to_result: Callable[[Any], Any]) -> Any` |
| `ml.connect.summarizer.summarize_dataframe` | Function | `(dataframe: Union[DataFrame, pd.DataFrame], column: str, metrics: List[str]) -> Dict[str, Any]` |
| `ml.connect.tuning.CrossValidator` | Class | `(estimator: Optional[Estimator], estimatorParamMaps: Optional[List[ParamMap]], evaluator: Optional[Evaluator], numFolds: int, seed: Optional[int], parallelism: int, foldCol: str)` |
| `ml.connect.tuning.CrossValidatorModel` | Class | `(bestModel: Optional[Model], avgMetrics: Optional[List[float]], stdMetrics: Optional[List[float]])` |
| `ml.connect.tuning.DataFrame` | Class | `(...)` |
| `ml.connect.tuning.Estimator` | Class | `(...)` |
| `ml.connect.tuning.Evaluator` | Class | `(...)` |
| `ml.connect.tuning.HasParallelism` | Class | `()` |
| `ml.connect.tuning.HasSeed` | Class | `()` |
| `ml.connect.tuning.MetaAlgorithmReadWrite` | Class | `(...)` |
| `ml.connect.tuning.Model` | Class | `(...)` |
| `ml.connect.tuning.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.connect.tuning.ParamMap` | Object | `` |
| `ml.connect.tuning.Params` | Class | `()` |
| `ml.connect.tuning.ParamsReadWrite` | Class | `(...)` |
| `ml.connect.tuning.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.connect.tuning.TypeConverters` | Class | `(...)` |
| `ml.connect.tuning.col` | Function | `(col: str) -> Column` |
| `ml.connect.tuning.inheritable_thread_target` | Function | `(f: Optional[Union[Callable, SparkSession]]) -> Callable` |
| `ml.connect.tuning.is_remote` | Function | `() -> bool` |
| `ml.connect.tuning.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.connect.tuning.lit` | Function | `(col: Any) -> Column` |
| `ml.connect.tuning.rand` | Function | `(seed: Optional[int]) -> Column` |
| `ml.connect.tuning.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.connect.util.DataFrame` | Class | `(...)` |
| `ml.connect.util.FuncT` | Object | `` |
| `ml.connect.util.aggregate_dataframe` | Function | `(dataframe: Union[DataFrame, pd.DataFrame], input_col_names: List[str], local_agg_fn: Callable[[pd.DataFrame], Any], merge_agg_state: Callable[[Any, Any], Any], agg_state_to_result: Callable[[Any], Any]) -> Any` |
| `ml.connect.util.cloudpickle` | Object | `` |
| `ml.connect.util.col` | Function | `(col: str) -> Column` |
| `ml.connect.util.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `ml.connect.util.pb2` | Object | `` |
| `ml.connect.util.transform_dataframe_column` | Function | `(dataframe: Union[DataFrame, pd.DataFrame], input_cols: List[str], transform_fn: Callable[..., Any], output_cols: List[Tuple[str, str]]) -> Union[DataFrame, pd.DataFrame]` |
| `ml.deepspeed.deepspeed_distributor.DeepspeedTorchDistributor` | Class | `(numGpus: int, nnodes: int, localMode: bool, useGpu: bool, deepspeedConfig: Optional[Union[str, Dict[str, Any]]])` |
| `ml.deepspeed.deepspeed_distributor.TorchDistributor` | Class | `(num_processes: int, local_mode: bool, use_gpu: bool, _ssl_conf: str)` |
| `ml.dl_util.FunctionPickler` | Class | `(...)` |
| `ml.dl_util.cloudpickle` | Object | `` |
| `ml.evaluation.BinaryClassificationEvaluator` | Class | `(rawPredictionCol: str, labelCol: str, metricName: BinaryClassificationEvaluatorMetricType, weightCol: Optional[str], numBins: int)` |
| `ml.evaluation.BinaryClassificationEvaluatorMetricType` | Object | `` |
| `ml.evaluation.ClusteringEvaluator` | Class | `(predictionCol: str, featuresCol: str, metricName: ClusteringEvaluatorMetricType, distanceMeasure: str, weightCol: Optional[str])` |
| `ml.evaluation.ClusteringEvaluatorDistanceMeasureType` | Object | `` |
| `ml.evaluation.ClusteringEvaluatorMetricType` | Object | `` |
| `ml.evaluation.DataFrame` | Class | `(...)` |
| `ml.evaluation.Evaluator` | Class | `(...)` |
| `ml.evaluation.HasFeaturesCol` | Class | `()` |
| `ml.evaluation.HasLabelCol` | Class | `()` |
| `ml.evaluation.HasPredictionCol` | Class | `()` |
| `ml.evaluation.HasProbabilityCol` | Class | `()` |
| `ml.evaluation.HasRawPredictionCol` | Class | `()` |
| `ml.evaluation.HasWeightCol` | Class | `()` |
| `ml.evaluation.JavaEvaluator` | Class | `(...)` |
| `ml.evaluation.JavaMLReadable` | Class | `(...)` |
| `ml.evaluation.JavaMLWritable` | Class | `(...)` |
| `ml.evaluation.JavaParams` | Class | `(...)` |
| `ml.evaluation.MulticlassClassificationEvaluator` | Class | `(predictionCol: str, labelCol: str, metricName: MulticlassClassificationEvaluatorMetricType, weightCol: Optional[str], metricLabel: float, beta: float, probabilityCol: str, eps: float)` |
| `ml.evaluation.MulticlassClassificationEvaluatorMetricType` | Object | `` |
| `ml.evaluation.MultilabelClassificationEvaluator` | Class | `(predictionCol: str, labelCol: str, metricName: MultilabelClassificationEvaluatorMetricType, metricLabel: float)` |
| `ml.evaluation.MultilabelClassificationEvaluatorMetricType` | Object | `` |
| `ml.evaluation.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.evaluation.ParamMap` | Object | `` |
| `ml.evaluation.Params` | Class | `()` |
| `ml.evaluation.RankingEvaluator` | Class | `(predictionCol: str, labelCol: str, metricName: RankingEvaluatorMetricType, k: int)` |
| `ml.evaluation.RankingEvaluatorMetricType` | Object | `` |
| `ml.evaluation.RegressionEvaluator` | Class | `(predictionCol: str, labelCol: str, metricName: RegressionEvaluatorMetricType, weightCol: Optional[str], throughOrigin: bool)` |
| `ml.evaluation.RegressionEvaluatorMetricType` | Object | `` |
| `ml.evaluation.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.evaluation.TypeConverters` | Class | `(...)` |
| `ml.evaluation.globs` | Object | `` |
| `ml.evaluation.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.evaluation.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.evaluation.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.evaluation.spark` | Object | `` |
| `ml.evaluation.temp_path` | Object | `` |
| `ml.evaluation.try_remote_evaluate` | Function | `(f: FuncT) -> FuncT` |
| `ml.feature.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `ml.feature.Binarizer` | Class | `(threshold: float, inputCol: Optional[str], outputCol: Optional[str], thresholds: Optional[List[float]], inputCols: Optional[List[str]], outputCols: Optional[List[str]])` |
| `ml.feature.BucketedRandomProjectionLSH` | Class | `(inputCol: Optional[str], outputCol: Optional[str], seed: Optional[int], numHashTables: int, bucketLength: Optional[float])` |
| `ml.feature.BucketedRandomProjectionLSHModel` | Class | `(...)` |
| `ml.feature.Bucketizer` | Class | `(splits: Optional[List[float]], inputCol: Optional[str], outputCol: Optional[str], handleInvalid: str, splitsArray: Optional[List[List[float]]], inputCols: Optional[List[str]], outputCols: Optional[List[str]])` |
| `ml.feature.ChiSqSelector` | Class | `(numTopFeatures: int, featuresCol: str, outputCol: Optional[str], labelCol: str, selectorType: str, percentile: float, fpr: float, fdr: float, fwe: float)` |
| `ml.feature.ChiSqSelectorModel` | Class | `(...)` |
| `ml.feature.CountVectorizer` | Class | `(minTF: float, minDF: float, maxDF: float, vocabSize: int, binary: bool, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.CountVectorizerModel` | Class | `(...)` |
| `ml.feature.DCT` | Class | `(inverse: bool, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.DataFrame` | Class | `(...)` |
| `ml.feature.DenseMatrix` | Class | `(numRows: int, numCols: int, values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `ml.feature.DenseVector` | Class | `(ar: Union[bytes, np.ndarray, Iterable[float]])` |
| `ml.feature.ElementwiseProduct` | Class | `(scalingVec: Optional[Vector], inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.FeatureHasher` | Class | `(numFeatures: int, inputCols: Optional[List[str]], outputCol: Optional[str], categoricalCols: Optional[List[str]])` |
| `ml.feature.HasFeaturesCol` | Class | `()` |
| `ml.feature.HasHandleInvalid` | Class | `()` |
| `ml.feature.HasInputCol` | Class | `()` |
| `ml.feature.HasInputCols` | Class | `()` |
| `ml.feature.HasLabelCol` | Class | `()` |
| `ml.feature.HasMaxIter` | Class | `()` |
| `ml.feature.HasNumFeatures` | Class | `()` |
| `ml.feature.HasOutputCol` | Class | `()` |
| `ml.feature.HasOutputCols` | Class | `()` |
| `ml.feature.HasRelativeError` | Class | `()` |
| `ml.feature.HasSeed` | Class | `()` |
| `ml.feature.HasStepSize` | Class | `()` |
| `ml.feature.HasThreshold` | Class | `()` |
| `ml.feature.HasThresholds` | Class | `()` |
| `ml.feature.HashingTF` | Class | `(numFeatures: int, binary: bool, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.IDF` | Class | `(minDocFreq: int, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.IDFModel` | Class | `(...)` |
| `ml.feature.Imputer` | Class | `(strategy: str, missingValue: float, inputCols: Optional[List[str]], outputCols: Optional[List[str]], inputCol: Optional[str], outputCol: Optional[str], relativeError: float)` |
| `ml.feature.ImputerModel` | Class | `(...)` |
| `ml.feature.IndexToString` | Class | `(inputCol: Optional[str], outputCol: Optional[str], labels: Optional[List[str]])` |
| `ml.feature.Interaction` | Class | `(inputCols: Optional[List[str]], outputCol: Optional[str])` |
| `ml.feature.JM` | Object | `` |
| `ml.feature.JavaEstimator` | Class | `(...)` |
| `ml.feature.JavaMLReadable` | Class | `(...)` |
| `ml.feature.JavaMLWritable` | Class | `(...)` |
| `ml.feature.JavaModel` | Class | `(java_model: Optional[JavaObject])` |
| `ml.feature.JavaParams` | Class | `(...)` |
| `ml.feature.JavaTransformer` | Class | `(...)` |
| `ml.feature.MaxAbsScaler` | Class | `(inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.MaxAbsScalerModel` | Class | `(...)` |
| `ml.feature.MinHashLSH` | Class | `(inputCol: Optional[str], outputCol: Optional[str], seed: Optional[int], numHashTables: int)` |
| `ml.feature.MinHashLSHModel` | Class | `(...)` |
| `ml.feature.MinMaxScaler` | Class | `(min: float, max: float, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.MinMaxScalerModel` | Class | `(...)` |
| `ml.feature.NGram` | Class | `(n: int, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.Normalizer` | Class | `(p: float, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.OneHotEncoder` | Class | `(inputCols: Optional[List[str]], outputCols: Optional[List[str]], handleInvalid: str, dropLast: bool, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.OneHotEncoderModel` | Class | `(...)` |
| `ml.feature.P` | Object | `` |
| `ml.feature.PCA` | Class | `(k: Optional[int], inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.PCAModel` | Class | `(...)` |
| `ml.feature.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.feature.Params` | Class | `()` |
| `ml.feature.PolynomialExpansion` | Class | `(degree: int, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.QuantileDiscretizer` | Class | `(numBuckets: int, inputCol: Optional[str], outputCol: Optional[str], relativeError: float, handleInvalid: str, numBucketsArray: Optional[List[int]], inputCols: Optional[List[str]], outputCols: Optional[List[str]])` |
| `ml.feature.RFormula` | Class | `(formula: Optional[str], featuresCol: str, labelCol: str, forceIndexLabel: bool, stringIndexerOrderType: str, handleInvalid: str)` |
| `ml.feature.RFormulaModel` | Class | `(...)` |
| `ml.feature.RegexTokenizer` | Class | `(minTokenLength: int, gaps: bool, pattern: str, inputCol: Optional[str], outputCol: Optional[str], toLowercase: bool)` |
| `ml.feature.RemoteModelRef` | Class | `(ref_id: str)` |
| `ml.feature.RobustScaler` | Class | `(lower: float, upper: float, withCentering: bool, withScaling: bool, inputCol: Optional[str], outputCol: Optional[str], relativeError: float)` |
| `ml.feature.RobustScalerModel` | Class | `(...)` |
| `ml.feature.Row` | Class | `(...)` |
| `ml.feature.SQLTransformer` | Class | `(statement: Optional[str])` |
| `ml.feature.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.feature.StandardScaler` | Class | `(withMean: bool, withStd: bool, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.StandardScalerModel` | Class | `(...)` |
| `ml.feature.StopWordsRemover` | Class | `(inputCol: Optional[str], outputCol: Optional[str], stopWords: Optional[List[str]], caseSensitive: bool, locale: Optional[str], inputCols: Optional[List[str]], outputCols: Optional[List[str]])` |
| `ml.feature.StringIndexer` | Class | `(inputCol: Optional[str], outputCol: Optional[str], inputCols: Optional[List[str]], outputCols: Optional[List[str]], handleInvalid: str, stringOrderType: str)` |
| `ml.feature.StringIndexerModel` | Class | `(...)` |
| `ml.feature.StringType` | Class | `(collation: str)` |
| `ml.feature.TargetEncoder` | Class | `(inputCols: Optional[List[str]], outputCols: Optional[List[str]], labelCol: str, handleInvalid: str, targetType: str, smoothing: float, inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.TargetEncoderModel` | Class | `(...)` |
| `ml.feature.Tokenizer` | Class | `(inputCol: Optional[str], outputCol: Optional[str])` |
| `ml.feature.TypeConverters` | Class | `(...)` |
| `ml.feature.UnivariateFeatureSelector` | Class | `(featuresCol: str, outputCol: Optional[str], labelCol: str, selectionMode: str)` |
| `ml.feature.UnivariateFeatureSelectorModel` | Class | `(...)` |
| `ml.feature.VarianceThresholdSelector` | Class | `(featuresCol: str, outputCol: Optional[str], varianceThreshold: float)` |
| `ml.feature.VarianceThresholdSelectorModel` | Class | `(...)` |
| `ml.feature.Vector` | Class | `(...)` |
| `ml.feature.VectorAssembler` | Class | `(inputCols: Optional[List[str]], outputCol: Optional[str], handleInvalid: str)` |
| `ml.feature.VectorIndexer` | Class | `(maxCategories: int, inputCol: Optional[str], outputCol: Optional[str], handleInvalid: str)` |
| `ml.feature.VectorIndexerModel` | Class | `(...)` |
| `ml.feature.VectorSizeHint` | Class | `(inputCol: Optional[str], size: Optional[int], handleInvalid: str)` |
| `ml.feature.VectorSlicer` | Class | `(inputCol: Optional[str], outputCol: Optional[str], indices: Optional[List[int]], names: Optional[List[str]])` |
| `ml.feature.Word2Vec` | Class | `(vectorSize: int, minCount: int, numPartitions: int, stepSize: float, maxIter: int, seed: Optional[int], inputCol: Optional[str], outputCol: Optional[str], windowSize: int, maxSentenceLength: int)` |
| `ml.feature.Word2VecModel` | Class | `(...)` |
| `ml.feature.features` | Object | `` |
| `ml.feature.globs` | Object | `` |
| `ml.feature.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.feature.invoke_helper_attr` | Function | `(method: str, args: Any) -> Any` |
| `ml.feature.is_remote` | Function | `() -> bool` |
| `ml.feature.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.feature.sc` | Object | `` |
| `ml.feature.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.feature.spark` | Object | `` |
| `ml.feature.temp_path` | Object | `` |
| `ml.feature.testData` | Object | `` |
| `ml.feature.try_remote_attribute_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.fpm.DataFrame` | Class | `(...)` |
| `ml.fpm.FPGrowth` | Class | `(minSupport: float, minConfidence: float, itemsCol: str, predictionCol: str, numPartitions: Optional[int])` |
| `ml.fpm.FPGrowthModel` | Class | `(...)` |
| `ml.fpm.HasPredictionCol` | Class | `()` |
| `ml.fpm.JavaEstimator` | Class | `(...)` |
| `ml.fpm.JavaMLReadable` | Class | `(...)` |
| `ml.fpm.JavaMLWritable` | Class | `(...)` |
| `ml.fpm.JavaModel` | Class | `(java_model: Optional[JavaObject])` |
| `ml.fpm.JavaParams` | Class | `(...)` |
| `ml.fpm.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.fpm.Params` | Class | `()` |
| `ml.fpm.PrefixSpan` | Class | `(minSupport: float, maxPatternLength: int, maxLocalProjDBSize: int, sequenceCol: str)` |
| `ml.fpm.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.fpm.TypeConverters` | Class | `(...)` |
| `ml.fpm.globs` | Object | `` |
| `ml.fpm.invoke_helper_relation` | Function | `(method: str, args: Any) -> ConnectDataFrame` |
| `ml.fpm.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.fpm.sc` | Object | `` |
| `ml.fpm.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.fpm.spark` | Object | `` |
| `ml.fpm.temp_path` | Object | `` |
| `ml.fpm.try_remote_attribute_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.functions.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `ml.functions.ByteType` | Class | `(...)` |
| `ml.functions.Column` | Class | `(...)` |
| `ml.functions.DataType` | Class | `(...)` |
| `ml.functions.DoubleType` | Class | `(...)` |
| `ml.functions.FloatType` | Class | `(...)` |
| `ml.functions.IntegerType` | Class | `(...)` |
| `ml.functions.LongType` | Class | `(...)` |
| `ml.functions.PredictBatchFunction` | Object | `` |
| `ml.functions.ShortType` | Class | `(...)` |
| `ml.functions.StringType` | Class | `(collation: str)` |
| `ml.functions.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `ml.functions.UserDefinedFunctionLike` | Class | `(...)` |
| `ml.functions.array_to_vector` | Function | `(col: Column) -> Column` |
| `ml.functions.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `ml.functions.predict_batch_udf` | Function | `(make_predict_fn: Callable[[], PredictBatchFunction], return_type: DataType, batch_size: int, input_tensor_shapes: Optional[Union[List[Optional[List[int]]], Mapping[int, List[int]]]]) -> UserDefinedFunctionLike` |
| `ml.functions.supported_scalar_types` | Object | `` |
| `ml.functions.try_remote_functions` | Function | `(f: FuncT) -> FuncT` |
| `ml.functions.vector_to_array` | Function | `(col: Column, dtype: str) -> Column` |
| `ml.image.ImageSchema` | Object | `` |
| `ml.image.Row` | Class | `(...)` |
| `ml.image.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.image.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `ml.linalg.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `ml.linalg.BooleanType` | Class | `(...)` |
| `ml.linalg.ByteType` | Class | `(...)` |
| `ml.linalg.DenseMatrix` | Class | `(numRows: int, numCols: int, values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `ml.linalg.DenseVector` | Class | `(ar: Union[bytes, np.ndarray, Iterable[float]])` |
| `ml.linalg.DoubleType` | Class | `(...)` |
| `ml.linalg.IntegerType` | Class | `(...)` |
| `ml.linalg.Matrices` | Class | `(...)` |
| `ml.linalg.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `ml.linalg.MatrixUDT` | Class | `(...)` |
| `ml.linalg.NormType` | Object | `` |
| `ml.linalg.SparseMatrix` | Class | `(numRows: int, numCols: int, colPtrs: Union[bytes, Iterable[int]], rowIndices: Union[bytes, Iterable[int]], values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `ml.linalg.SparseVector` | Class | `(size: int, args: Union[bytes, Tuple[int, float], Iterable[float], Iterable[Tuple[int, float]], Dict[int, float]])` |
| `ml.linalg.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `ml.linalg.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `ml.linalg.UserDefinedType` | Class | `(...)` |
| `ml.linalg.Vector` | Class | `(...)` |
| `ml.linalg.VectorLike` | Object | `` |
| `ml.linalg.VectorUDT` | Class | `(...)` |
| `ml.linalg.Vectors` | Class | `(...)` |
| `ml.model_cache.ModelCache` | Class | `(...)` |
| `ml.param.DenseVector` | Class | `(ar: Union[bytes, np.ndarray, Iterable[float]])` |
| `ml.param.Identifiable` | Class | `()` |
| `ml.param.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `ml.param.P` | Object | `` |
| `ml.param.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.param.ParamMap` | Object | `` |
| `ml.param.Params` | Class | `()` |
| `ml.param.T` | Object | `` |
| `ml.param.TypeConverters` | Class | `(...)` |
| `ml.param.Vector` | Class | `(...)` |
| `ml.param.is_remote_only` | Function | `() -> bool` |
| `ml.param.shared.HasAggregationDepth` | Class | `()` |
| `ml.param.shared.HasBatchSize` | Class | `()` |
| `ml.param.shared.HasBlockSize` | Class | `()` |
| `ml.param.shared.HasCheckpointInterval` | Class | `()` |
| `ml.param.shared.HasCollectSubModels` | Class | `()` |
| `ml.param.shared.HasDistanceMeasure` | Class | `()` |
| `ml.param.shared.HasElasticNetParam` | Class | `()` |
| `ml.param.shared.HasFeatureSizes` | Class | `()` |
| `ml.param.shared.HasFeaturesCol` | Class | `()` |
| `ml.param.shared.HasFitIntercept` | Class | `()` |
| `ml.param.shared.HasHandleInvalid` | Class | `()` |
| `ml.param.shared.HasInputCol` | Class | `()` |
| `ml.param.shared.HasInputCols` | Class | `()` |
| `ml.param.shared.HasLabelCol` | Class | `()` |
| `ml.param.shared.HasLearningRate` | Class | `()` |
| `ml.param.shared.HasLoss` | Class | `()` |
| `ml.param.shared.HasMaxBlockSizeInMB` | Class | `()` |
| `ml.param.shared.HasMaxIter` | Class | `()` |
| `ml.param.shared.HasMomentum` | Class | `()` |
| `ml.param.shared.HasNumFeatures` | Class | `()` |
| `ml.param.shared.HasNumTrainWorkers` | Class | `()` |
| `ml.param.shared.HasOutputCol` | Class | `()` |
| `ml.param.shared.HasOutputCols` | Class | `()` |
| `ml.param.shared.HasParallelism` | Class | `()` |
| `ml.param.shared.HasPredictionCol` | Class | `()` |
| `ml.param.shared.HasProbabilityCol` | Class | `()` |
| `ml.param.shared.HasRawPredictionCol` | Class | `()` |
| `ml.param.shared.HasRegParam` | Class | `()` |
| `ml.param.shared.HasRelativeError` | Class | `()` |
| `ml.param.shared.HasSeed` | Class | `()` |
| `ml.param.shared.HasSolver` | Class | `()` |
| `ml.param.shared.HasStandardization` | Class | `()` |
| `ml.param.shared.HasStepSize` | Class | `()` |
| `ml.param.shared.HasThreshold` | Class | `()` |
| `ml.param.shared.HasThresholds` | Class | `()` |
| `ml.param.shared.HasTol` | Class | `()` |
| `ml.param.shared.HasValidationIndicatorCol` | Class | `()` |
| `ml.param.shared.HasVarianceCol` | Class | `()` |
| `ml.param.shared.HasWeightCol` | Class | `()` |
| `ml.param.shared.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.param.shared.Params` | Class | `()` |
| `ml.param.shared.TypeConverters` | Class | `(...)` |
| `ml.pipeline.DataFrame` | Class | `(...)` |
| `ml.pipeline.DefaultParamsReader` | Class | `(cls: Type[DefaultParamsReadable[RL]])` |
| `ml.pipeline.DefaultParamsWriter` | Class | `(instance: Params)` |
| `ml.pipeline.Estimator` | Class | `(...)` |
| `ml.pipeline.JavaMLWritable` | Class | `(...)` |
| `ml.pipeline.JavaMLWriter` | Class | `(instance: JavaMLWritable)` |
| `ml.pipeline.JavaParams` | Class | `(...)` |
| `ml.pipeline.MLReadable` | Class | `(...)` |
| `ml.pipeline.MLReader` | Class | `()` |
| `ml.pipeline.MLWritable` | Class | `(...)` |
| `ml.pipeline.MLWriter` | Class | `()` |
| `ml.pipeline.Model` | Class | `(...)` |
| `ml.pipeline.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.pipeline.ParamMap` | Object | `` |
| `ml.pipeline.Params` | Class | `()` |
| `ml.pipeline.Pipeline` | Class | `(stages: Optional[List[PipelineStage]])` |
| `ml.pipeline.PipelineModel` | Class | `(stages: List[Transformer])` |
| `ml.pipeline.PipelineModelReader` | Class | `(cls: Type[PipelineModel])` |
| `ml.pipeline.PipelineModelWriter` | Class | `(instance: PipelineModel)` |
| `ml.pipeline.PipelineReader` | Class | `(cls: Type[Pipeline])` |
| `ml.pipeline.PipelineSharedReadWrite` | Class | `(...)` |
| `ml.pipeline.PipelineStage` | Object | `` |
| `ml.pipeline.PipelineWriter` | Class | `(instance: Pipeline)` |
| `ml.pipeline.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `ml.pipeline.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.pipeline.Transformer` | Class | `(...)` |
| `ml.pipeline.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.pipeline.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.pipeline.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.pipeline.try_remote_read` | Function | `(f: FuncT) -> FuncT` |
| `ml.pipeline.try_remote_write` | Function | `(f: FuncT) -> FuncT` |
| `ml.recommendation.ALS` | Class | `(rank: int, maxIter: int, regParam: float, numUserBlocks: int, numItemBlocks: int, implicitPrefs: bool, alpha: float, userCol: str, itemCol: str, seed: Optional[int], ratingCol: str, nonnegative: bool, checkpointInterval: int, intermediateStorageLevel: str, finalStorageLevel: str, coldStartStrategy: str, blockSize: int)` |
| `ml.recommendation.ALSModel` | Class | `(...)` |
| `ml.recommendation.DataFrame` | Class | `(...)` |
| `ml.recommendation.HasBlockSize` | Class | `()` |
| `ml.recommendation.HasCheckpointInterval` | Class | `()` |
| `ml.recommendation.HasMaxIter` | Class | `()` |
| `ml.recommendation.HasPredictionCol` | Class | `()` |
| `ml.recommendation.HasRegParam` | Class | `()` |
| `ml.recommendation.HasSeed` | Class | `()` |
| `ml.recommendation.JavaEstimator` | Class | `(...)` |
| `ml.recommendation.JavaMLReadable` | Class | `(...)` |
| `ml.recommendation.JavaMLWritable` | Class | `(...)` |
| `ml.recommendation.JavaModel` | Class | `(java_model: Optional[JavaObject])` |
| `ml.recommendation.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.recommendation.Params` | Class | `()` |
| `ml.recommendation.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.recommendation.TypeConverters` | Class | `(...)` |
| `ml.recommendation.globs` | Object | `` |
| `ml.recommendation.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.recommendation.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.recommendation.sc` | Object | `` |
| `ml.recommendation.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.recommendation.spark` | Object | `` |
| `ml.recommendation.temp_path` | Object | `` |
| `ml.recommendation.try_remote_attribute_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.regression.AFTSurvivalRegression` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, fitIntercept: bool, maxIter: int, tol: float, censorCol: str, quantileProbabilities: List[float], quantilesCol: Optional[str], aggregationDepth: int, maxBlockSizeInMB: float)` |
| `ml.regression.AFTSurvivalRegressionModel` | Class | `(...)` |
| `ml.regression.DataFrame` | Class | `(...)` |
| `ml.regression.DecisionTreeRegressionModel` | Class | `(...)` |
| `ml.regression.DecisionTreeRegressor` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxDepth: int, maxBins: int, minInstancesPerNode: int, minInfoGain: float, maxMemoryInMB: int, cacheNodeIds: bool, checkpointInterval: int, impurity: str, seed: Optional[int], varianceCol: Optional[str], weightCol: Optional[str], leafCol: str, minWeightFractionPerNode: float)` |
| `ml.regression.FMRegressionModel` | Class | `(...)` |
| `ml.regression.FMRegressor` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, factorSize: int, fitIntercept: bool, fitLinear: bool, regParam: float, miniBatchFraction: float, initStd: float, maxIter: int, stepSize: float, tol: float, solver: str, seed: Optional[int])` |
| `ml.regression.GBTRegressionModel` | Class | `(...)` |
| `ml.regression.GBTRegressor` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxDepth: int, maxBins: int, minInstancesPerNode: int, minInfoGain: float, maxMemoryInMB: int, cacheNodeIds: bool, subsamplingRate: float, checkpointInterval: int, lossType: str, maxIter: int, stepSize: float, seed: Optional[int], impurity: str, featureSubsetStrategy: str, validationTol: float, validationIndicatorCol: Optional[str], leafCol: str, minWeightFractionPerNode: float, weightCol: Optional[str])` |
| `ml.regression.GeneralJavaMLWritable` | Class | `(...)` |
| `ml.regression.GeneralizedLinearRegression` | Class | `(labelCol: str, featuresCol: str, predictionCol: str, family: str, link: Optional[str], fitIntercept: bool, maxIter: int, tol: float, regParam: float, weightCol: Optional[str], solver: str, linkPredictionCol: Optional[str], variancePower: float, linkPower: Optional[float], offsetCol: Optional[str], aggregationDepth: int)` |
| `ml.regression.GeneralizedLinearRegressionModel` | Class | `(...)` |
| `ml.regression.GeneralizedLinearRegressionSummary` | Class | `(...)` |
| `ml.regression.GeneralizedLinearRegressionTrainingSummary` | Class | `(...)` |
| `ml.regression.HasAggregationDepth` | Class | `()` |
| `ml.regression.HasElasticNetParam` | Class | `()` |
| `ml.regression.HasFeaturesCol` | Class | `()` |
| `ml.regression.HasFitIntercept` | Class | `()` |
| `ml.regression.HasLabelCol` | Class | `()` |
| `ml.regression.HasLoss` | Class | `()` |
| `ml.regression.HasMaxBlockSizeInMB` | Class | `()` |
| `ml.regression.HasMaxIter` | Class | `()` |
| `ml.regression.HasPredictionCol` | Class | `()` |
| `ml.regression.HasRegParam` | Class | `()` |
| `ml.regression.HasSeed` | Class | `()` |
| `ml.regression.HasSolver` | Class | `()` |
| `ml.regression.HasStandardization` | Class | `()` |
| `ml.regression.HasStepSize` | Class | `()` |
| `ml.regression.HasTol` | Class | `()` |
| `ml.regression.HasTrainingSummary` | Class | `(...)` |
| `ml.regression.HasVarianceCol` | Class | `()` |
| `ml.regression.HasWeightCol` | Class | `()` |
| `ml.regression.IsotonicRegression` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, weightCol: Optional[str], isotonic: bool, featureIndex: int)` |
| `ml.regression.IsotonicRegressionModel` | Class | `(...)` |
| `ml.regression.JM` | Object | `` |
| `ml.regression.JavaEstimator` | Class | `(...)` |
| `ml.regression.JavaMLReadable` | Class | `(...)` |
| `ml.regression.JavaMLWritable` | Class | `(...)` |
| `ml.regression.JavaModel` | Class | `(java_model: Optional[JavaObject])` |
| `ml.regression.JavaPredictionModel` | Class | `(...)` |
| `ml.regression.JavaPredictor` | Class | `(...)` |
| `ml.regression.JavaTransformer` | Class | `(...)` |
| `ml.regression.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.regression.LinearRegression` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxIter: int, regParam: float, elasticNetParam: float, tol: float, fitIntercept: bool, standardization: bool, solver: str, weightCol: Optional[str], aggregationDepth: int, loss: str, epsilon: float, maxBlockSizeInMB: float)` |
| `ml.regression.LinearRegressionModel` | Class | `(...)` |
| `ml.regression.LinearRegressionSummary` | Class | `(...)` |
| `ml.regression.LinearRegressionTrainingSummary` | Class | `(...)` |
| `ml.regression.M` | Object | `` |
| `ml.regression.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `ml.regression.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.regression.Params` | Class | `()` |
| `ml.regression.PredictionModel` | Class | `(...)` |
| `ml.regression.Predictor` | Class | `(...)` |
| `ml.regression.RandomForestRegressionModel` | Class | `(...)` |
| `ml.regression.RandomForestRegressor` | Class | `(featuresCol: str, labelCol: str, predictionCol: str, maxDepth: int, maxBins: int, minInstancesPerNode: int, minInfoGain: float, maxMemoryInMB: int, cacheNodeIds: bool, checkpointInterval: int, impurity: str, subsamplingRate: float, seed: Optional[int], numTrees: int, featureSubsetStrategy: str, leafCol: str, minWeightFractionPerNode: float, weightCol: Optional[str], bootstrap: Optional[bool])` |
| `ml.regression.RegressionModel` | Class | `(...)` |
| `ml.regression.Regressor` | Class | `(...)` |
| `ml.regression.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.regression.T` | Object | `` |
| `ml.regression.Transformer` | Class | `(...)` |
| `ml.regression.TypeConverters` | Class | `(...)` |
| `ml.regression.Vector` | Class | `(...)` |
| `ml.regression.globs` | Object | `` |
| `ml.regression.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.regression.is_remote` | Function | `() -> bool` |
| `ml.regression.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.regression.sc` | Object | `` |
| `ml.regression.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.regression.spark` | Object | `` |
| `ml.regression.temp_path` | Object | `` |
| `ml.regression.try_remote_attribute_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.stat.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `ml.stat.ChiSquareTest` | Class | `(...)` |
| `ml.stat.Column` | Class | `(...)` |
| `ml.stat.Correlation` | Class | `(...)` |
| `ml.stat.DataFrame` | Class | `(...)` |
| `ml.stat.DoubleType` | Class | `(...)` |
| `ml.stat.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.stat.KolmogorovSmirnovTest` | Class | `(...)` |
| `ml.stat.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `ml.stat.MultivariateGaussian` | Class | `(mean: Vector, cov: Matrix)` |
| `ml.stat.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.stat.Summarizer` | Class | `(...)` |
| `ml.stat.SummaryBuilder` | Class | `(jSummaryBuilder: JavaObject)` |
| `ml.stat.Vector` | Class | `(...)` |
| `ml.stat.globs` | Object | `` |
| `ml.stat.invoke_helper_relation` | Function | `(method: str, args: Any) -> ConnectDataFrame` |
| `ml.stat.is_remote` | Function | `() -> bool` |
| `ml.stat.lit` | Function | `(col: Any) -> Column` |
| `ml.stat.sc` | Object | `` |
| `ml.stat.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.stat.spark` | Object | `` |
| `ml.torch.data.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `ml.torch.distributor.BarrierTaskContext` | Class | `(...)` |
| `ml.torch.distributor.DataFrame` | Class | `(...)` |
| `ml.torch.distributor.Distributor` | Class | `(num_processes: int, local_mode: bool, use_gpu: bool, ssl_conf: Optional[str])` |
| `ml.torch.distributor.FunctionPickler` | Class | `(...)` |
| `ml.torch.distributor.LogStreamingClient` | Class | `(address: str, port: int, timeout: int)` |
| `ml.torch.distributor.LogStreamingServer` | Class | `()` |
| `ml.torch.distributor.ResourceInformation` | Class | `(name: str, addresses: List[str])` |
| `ml.torch.distributor.SPARK_DATAFRAME_SCHEMA_FILE` | Object | `` |
| `ml.torch.distributor.SPARK_PARTITION_ARROW_DATA_FILE` | Object | `` |
| `ml.torch.distributor.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.torch.distributor.TorchDistributor` | Class | `(num_processes: int, local_mode: bool, use_gpu: bool, _ssl_conf: str)` |
| `ml.torch.distributor.cloudpickle` | Object | `` |
| `ml.torch.log_communication.LogStreamingClient` | Class | `(address: str, port: int, timeout: int)` |
| `ml.torch.log_communication.LogStreamingClientBase` | Class | `(...)` |
| `ml.torch.log_communication.LogStreamingServer` | Class | `()` |
| `ml.torch.log_communication.WriteLogToStdout` | Class | `(...)` |
| `ml.torch.torch_run_process_wrapper.args` | Object | `` |
| `ml.torch.torch_run_process_wrapper.check_parent_alive` | Function | `(task: subprocess.Popen) -> None` |
| `ml.torch.torch_run_process_wrapper.clean_and_terminate` | Function | `(task: subprocess.Popen) -> None` |
| `ml.torch.torch_run_process_wrapper.cmd` | Object | `` |
| `ml.torch.torch_run_process_wrapper.decoded` | Object | `` |
| `ml.torch.torch_run_process_wrapper.sigterm_handler` | Function | `(args: Any) -> None` |
| `ml.torch.torch_run_process_wrapper.t` | Object | `` |
| `ml.torch.torch_run_process_wrapper.task` | Object | `` |
| `ml.tree.HasCheckpointInterval` | Class | `()` |
| `ml.tree.HasMaxIter` | Class | `()` |
| `ml.tree.HasSeed` | Class | `()` |
| `ml.tree.HasStepSize` | Class | `()` |
| `ml.tree.HasValidationIndicatorCol` | Class | `()` |
| `ml.tree.HasWeightCol` | Class | `()` |
| `ml.tree.JavaPredictionModel` | Class | `(...)` |
| `ml.tree.P` | Object | `` |
| `ml.tree.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.tree.Params` | Class | `()` |
| `ml.tree.T` | Object | `` |
| `ml.tree.TypeConverters` | Class | `(...)` |
| `ml.tree.Vector` | Class | `(...)` |
| `ml.tree.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.tree.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.tuning.CrossValidator` | Class | `(estimator: Optional[Estimator], estimatorParamMaps: Optional[List[ParamMap]], evaluator: Optional[Evaluator], numFolds: int, seed: Optional[int], parallelism: int, collectSubModels: bool, foldCol: str)` |
| `ml.tuning.CrossValidatorModel` | Class | `(bestModel: Model, avgMetrics: Optional[List[float]], subModels: Optional[List[List[Model]]], stdMetrics: Optional[List[float]])` |
| `ml.tuning.CrossValidatorModelReader` | Class | `(cls: Type[CrossValidatorModel])` |
| `ml.tuning.CrossValidatorModelWriter` | Class | `(instance: CrossValidatorModel)` |
| `ml.tuning.CrossValidatorReader` | Class | `(cls: Type[CrossValidator])` |
| `ml.tuning.CrossValidatorWriter` | Class | `(instance: CrossValidator)` |
| `ml.tuning.DataFrame` | Class | `(...)` |
| `ml.tuning.DefaultParamsReader` | Class | `(cls: Type[DefaultParamsReadable[RL]])` |
| `ml.tuning.DefaultParamsWriter` | Class | `(instance: Params)` |
| `ml.tuning.Estimator` | Class | `(...)` |
| `ml.tuning.Evaluator` | Class | `(...)` |
| `ml.tuning.F` | Object | `` |
| `ml.tuning.HasCollectSubModels` | Class | `()` |
| `ml.tuning.HasParallelism` | Class | `()` |
| `ml.tuning.HasSeed` | Class | `()` |
| `ml.tuning.JavaEstimator` | Class | `(...)` |
| `ml.tuning.JavaEvaluator` | Class | `(...)` |
| `ml.tuning.JavaMLWriter` | Class | `(instance: JavaMLWritable)` |
| `ml.tuning.JavaParams` | Class | `(...)` |
| `ml.tuning.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.tuning.MLReadable` | Class | `(...)` |
| `ml.tuning.MLReader` | Class | `()` |
| `ml.tuning.MLWritable` | Class | `(...)` |
| `ml.tuning.MLWriter` | Class | `()` |
| `ml.tuning.MetaAlgorithmReadWrite` | Class | `(...)` |
| `ml.tuning.Model` | Class | `(...)` |
| `ml.tuning.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.tuning.ParamGridBuilder` | Class | `()` |
| `ml.tuning.ParamMap` | Object | `` |
| `ml.tuning.Params` | Class | `()` |
| `ml.tuning.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `ml.tuning.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.tuning.TrainValidationSplit` | Class | `(estimator: Optional[Estimator], estimatorParamMaps: Optional[List[ParamMap]], evaluator: Optional[Evaluator], trainRatio: float, parallelism: int, collectSubModels: bool, seed: Optional[int])` |
| `ml.tuning.TrainValidationSplitModel` | Class | `(bestModel: Model, validationMetrics: Optional[List[float]], subModels: Optional[List[Model]])` |
| `ml.tuning.TrainValidationSplitModelReader` | Class | `(cls: Type[TrainValidationSplitModel])` |
| `ml.tuning.TrainValidationSplitModelWriter` | Class | `(instance: TrainValidationSplitModel)` |
| `ml.tuning.TrainValidationSplitReader` | Class | `(cls: Type[TrainValidationSplit])` |
| `ml.tuning.TrainValidationSplitWriter` | Class | `(instance: TrainValidationSplit)` |
| `ml.tuning.Transformer` | Class | `(...)` |
| `ml.tuning.TypeConverters` | Class | `(...)` |
| `ml.tuning.globs` | Object | `` |
| `ml.tuning.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.tuning.inheritable_thread_target` | Function | `(f: Optional[Union[Callable, SparkSession]]) -> Callable` |
| `ml.tuning.keyword_only` | Function | `(func: _F) -> _F` |
| `ml.tuning.sc` | Object | `` |
| `ml.tuning.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.tuning.spark` | Object | `` |
| `ml.tuning.try_remote_read` | Function | `(f: FuncT) -> FuncT` |
| `ml.tuning.try_remote_write` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.BaseReadWrite` | Class | `()` |
| `ml.util.ConnectDataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `ml.util.DataFrame` | Class | `(...)` |
| `ml.util.DefaultParamsReadable` | Class | `(...)` |
| `ml.util.DefaultParamsReader` | Class | `(cls: Type[DefaultParamsReadable[RL]])` |
| `ml.util.DefaultParamsWritable` | Class | `(...)` |
| `ml.util.DefaultParamsWriter` | Class | `(instance: Params)` |
| `ml.util.FuncT` | Object | `` |
| `ml.util.GeneralJavaMLWritable` | Class | `(...)` |
| `ml.util.GeneralJavaMLWriter` | Class | `(instance: JavaMLWritable)` |
| `ml.util.GeneralMLWriter` | Class | `(...)` |
| `ml.util.HasTrainingSummary` | Class | `(...)` |
| `ml.util.Identifiable` | Class | `()` |
| `ml.util.JR` | Object | `` |
| `ml.util.JW` | Object | `` |
| `ml.util.JavaEstimator` | Class | `(...)` |
| `ml.util.JavaEvaluator` | Class | `(...)` |
| `ml.util.JavaMLReadable` | Class | `(...)` |
| `ml.util.JavaMLReader` | Class | `(clazz: Type[JavaMLReadable[RL]])` |
| `ml.util.JavaMLWritable` | Class | `(...)` |
| `ml.util.JavaMLWriter` | Class | `(instance: JavaMLWritable)` |
| `ml.util.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.util.MLReadable` | Class | `(...)` |
| `ml.util.MLReader` | Class | `()` |
| `ml.util.MLWritable` | Class | `(...)` |
| `ml.util.MLWriter` | Class | `()` |
| `ml.util.ML_CONNECT_HELPER_ID` | Object | `` |
| `ml.util.MetaAlgorithmReadWrite` | Class | `(...)` |
| `ml.util.Params` | Class | `()` |
| `ml.util.PipelineStage` | Object | `` |
| `ml.util.RL` | Object | `` |
| `ml.util.RW` | Object | `` |
| `ml.util.RemoteModelRef` | Class | `(ref_id: str)` |
| `ml.util.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `ml.util.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `ml.util.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `ml.util.T` | Object | `` |
| `ml.util.VersionUtils` | Class | `(...)` |
| `ml.util.W` | Object | `` |
| `ml.util.del_remote_cache` | Function | `(ref_id: str) -> None` |
| `ml.util.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.util.invoke_helper_attr` | Function | `(method: str, args: Any) -> Any` |
| `ml.util.invoke_helper_relation` | Function | `(method: str, args: Any) -> ConnectDataFrame` |
| `ml.util.invoke_remote_attribute_relation` | Function | `(instance: JavaWrapper, method: str, args: Any) -> ConnectDataFrame` |
| `ml.util.is_remote` | Function | `() -> bool` |
| `ml.util.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.util.try_remote_attribute_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_call` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_del` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_evaluate` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_fit` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_functions` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_intercept` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_not_supporting` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_read` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_return_java_class` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_transform_relation` | Function | `(f: FuncT) -> FuncT` |
| `ml.util.try_remote_write` | Function | `(f: FuncT) -> FuncT` |
| `ml.wrapper.DataFrame` | Class | `(...)` |
| `ml.wrapper.Estimator` | Class | `(...)` |
| `ml.wrapper.JM` | Object | `` |
| `ml.wrapper.JP` | Object | `` |
| `ml.wrapper.JW` | Object | `` |
| `ml.wrapper.JavaEstimator` | Class | `(...)` |
| `ml.wrapper.JavaModel` | Class | `(java_model: Optional[JavaObject])` |
| `ml.wrapper.JavaParams` | Class | `(...)` |
| `ml.wrapper.JavaPredictionModel` | Class | `(...)` |
| `ml.wrapper.JavaPredictor` | Class | `(...)` |
| `ml.wrapper.JavaTransformer` | Class | `(...)` |
| `ml.wrapper.JavaWrapper` | Class | `(java_obj: Optional[JavaObject])` |
| `ml.wrapper.Model` | Class | `(...)` |
| `ml.wrapper.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `ml.wrapper.ParamMap` | Object | `` |
| `ml.wrapper.Params` | Class | `()` |
| `ml.wrapper.PredictionModel` | Class | `(...)` |
| `ml.wrapper.Predictor` | Class | `(...)` |
| `ml.wrapper.T` | Object | `` |
| `ml.wrapper.Transformer` | Class | `(...)` |
| `ml.wrapper.inherit_doc` | Function | `(cls: C) -> C` |
| `ml.wrapper.is_remote` | Function | `() -> bool` |
| `ml.wrapper.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `ml.wrapper.try_remote_call` | Function | `(f: FuncT) -> FuncT` |
| `ml.wrapper.try_remote_del` | Function | `(f: FuncT) -> FuncT` |
| `ml.wrapper.try_remote_fit` | Function | `(f: FuncT) -> FuncT` |
| `ml.wrapper.try_remote_intercept` | Function | `(f: FuncT) -> FuncT` |
| `ml.wrapper.try_remote_return_java_class` | Function | `(f: FuncT) -> FuncT` |
| `ml.wrapper.try_remote_transform_relation` | Function | `(f: FuncT) -> FuncT` |
| `mllib.classification.DStream` | Class | `(jdstream: JavaObject, ssc: StreamingContext, jrdd_deserializer: Serializer)` |
| `mllib.classification.LabeledPoint` | Class | `(label: float, features: Iterable[float])` |
| `mllib.classification.LinearClassificationModel` | Class | `(weights: Vector, intercept: float)` |
| `mllib.classification.LinearModel` | Class | `(weights: Vector, intercept: float)` |
| `mllib.classification.Loader` | Class | `(...)` |
| `mllib.classification.LogisticRegressionModel` | Class | `(weights: Vector, intercept: float, numFeatures: int, numClasses: int)` |
| `mllib.classification.LogisticRegressionWithLBFGS` | Class | `(...)` |
| `mllib.classification.LogisticRegressionWithSGD` | Class | `(...)` |
| `mllib.classification.NaiveBayes` | Class | `(...)` |
| `mllib.classification.NaiveBayesModel` | Class | `(labels: numpy.ndarray, pi: numpy.ndarray, theta: numpy.ndarray)` |
| `mllib.classification.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.classification.SVMModel` | Class | `(weights: Vector, intercept: float)` |
| `mllib.classification.SVMWithSGD` | Class | `(...)` |
| `mllib.classification.Saveable` | Class | `(...)` |
| `mllib.classification.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.classification.StreamingLinearAlgorithm` | Class | `(model: Optional[LinearModel])` |
| `mllib.classification.StreamingLogisticRegressionWithSGD` | Class | `(stepSize: float, numIterations: int, miniBatchFraction: float, regParam: float, convergenceTol: float)` |
| `mllib.classification.Vector` | Class | `(...)` |
| `mllib.classification.VectorLike` | Object | `` |
| `mllib.classification.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.classification.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.classification.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.clustering.BisectingKMeans` | Class | `(...)` |
| `mllib.clustering.BisectingKMeansModel` | Class | `(java_model: JavaObject)` |
| `mllib.clustering.DStream` | Class | `(jdstream: JavaObject, ssc: StreamingContext, jrdd_deserializer: Serializer)` |
| `mllib.clustering.DenseVector` | Class | `(ar: Union[bytes, np.ndarray, Iterable[float]])` |
| `mllib.clustering.GaussianMixture` | Class | `(...)` |
| `mllib.clustering.GaussianMixtureModel` | Class | `(...)` |
| `mllib.clustering.JavaLoader` | Class | `(...)` |
| `mllib.clustering.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.clustering.JavaSaveable` | Class | `(...)` |
| `mllib.clustering.KMeans` | Class | `(...)` |
| `mllib.clustering.KMeansModel` | Class | `(centers: List[VectorLike])` |
| `mllib.clustering.LDA` | Class | `(...)` |
| `mllib.clustering.LDAModel` | Class | `(...)` |
| `mllib.clustering.Loader` | Class | `(...)` |
| `mllib.clustering.MultivariateGaussian` | Class | `(...)` |
| `mllib.clustering.PowerIterationClustering` | Class | `(...)` |
| `mllib.clustering.PowerIterationClusteringModel` | Class | `(...)` |
| `mllib.clustering.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.clustering.Saveable` | Class | `(...)` |
| `mllib.clustering.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.clustering.SparseVector` | Class | `(size: int, args: Union[bytes, Tuple[int, float], Iterable[float], Iterable[Tuple[int, float]], Dict[int, float]])` |
| `mllib.clustering.StreamingKMeans` | Class | `(k: int, decayFactor: float, timeUnit: str)` |
| `mllib.clustering.StreamingKMeansModel` | Class | `(clusterCenters: List[VectorLike], clusterWeights: VectorLike)` |
| `mllib.clustering.T` | Object | `` |
| `mllib.clustering.VectorLike` | Object | `` |
| `mllib.clustering.callJavaFunc` | Function | `(sc: pyspark.core.context.SparkContext, func: Callable[..., JavaObjectOrPickleDump], args: Any) -> Any` |
| `mllib.clustering.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.clustering.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.clustering.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.common.AutoBatchedSerializer` | Class | `(serializer, bestSize)` |
| `mllib.common.C` | Object | `` |
| `mllib.common.CPickleSerializer` | Object | `` |
| `mllib.common.DataFrame` | Class | `(...)` |
| `mllib.common.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.common.JavaObjectOrPickleDump` | Object | `` |
| `mllib.common.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.common.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.common.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `mllib.common.callJavaFunc` | Function | `(sc: pyspark.core.context.SparkContext, func: Callable[..., JavaObjectOrPickleDump], args: Any) -> Any` |
| `mllib.common.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.common.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.evaluation.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `mllib.evaluation.BinaryClassificationMetrics` | Class | `(scoreAndLabels: RDD[Tuple[float, float]])` |
| `mllib.evaluation.DoubleType` | Class | `(...)` |
| `mllib.evaluation.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.evaluation.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `mllib.evaluation.MulticlassMetrics` | Class | `(predictionAndLabels: RDD[Tuple[float, float]])` |
| `mllib.evaluation.MultilabelMetrics` | Class | `(predictionAndLabels: RDD[Tuple[List[float], List[float]]])` |
| `mllib.evaluation.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.evaluation.RankingMetrics` | Class | `(predictionAndLabels: Union[RDD[Tuple[List[T], List[T]]], RDD[Tuple[List[T], List[T], List[float]]]])` |
| `mllib.evaluation.RegressionMetrics` | Class | `(predictionAndObservations: RDD[Tuple[float, float]])` |
| `mllib.evaluation.SQLContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jsqlContext: Optional[JavaObject])` |
| `mllib.evaluation.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `mllib.evaluation.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `mllib.evaluation.T` | Object | `` |
| `mllib.evaluation.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.evaluation.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.feature.ChiSqSelector` | Class | `(numTopFeatures: int, selectorType: str, percentile: float, fpr: float, fdr: float, fwe: float)` |
| `mllib.feature.ChiSqSelectorModel` | Class | `(...)` |
| `mllib.feature.ElementwiseProduct` | Class | `(scalingVector: Vector)` |
| `mllib.feature.HashingTF` | Class | `(numFeatures: int)` |
| `mllib.feature.IDF` | Class | `(minDocFreq: int)` |
| `mllib.feature.IDFModel` | Class | `(...)` |
| `mllib.feature.JavaLoader` | Class | `(...)` |
| `mllib.feature.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.feature.JavaSaveable` | Class | `(...)` |
| `mllib.feature.JavaVectorTransformer` | Class | `(...)` |
| `mllib.feature.LabeledPoint` | Class | `(label: float, features: Iterable[float])` |
| `mllib.feature.Normalizer` | Class | `(p: float)` |
| `mllib.feature.PCA` | Class | `(k: int)` |
| `mllib.feature.PCAModel` | Class | `(...)` |
| `mllib.feature.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.feature.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.feature.StandardScaler` | Class | `(withMean: bool, withStd: bool)` |
| `mllib.feature.StandardScalerModel` | Class | `(...)` |
| `mllib.feature.Vector` | Class | `(...)` |
| `mllib.feature.VectorLike` | Object | `` |
| `mllib.feature.VectorTransformer` | Class | `(...)` |
| `mllib.feature.Vectors` | Class | `(...)` |
| `mllib.feature.Word2Vec` | Class | `()` |
| `mllib.feature.Word2VecModel` | Class | `(...)` |
| `mllib.feature.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.feature.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.fpm.FPGrowth` | Class | `(...)` |
| `mllib.fpm.FPGrowthModel` | Class | `(...)` |
| `mllib.fpm.JavaLoader` | Class | `(...)` |
| `mllib.fpm.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.fpm.JavaSaveable` | Class | `(...)` |
| `mllib.fpm.PrefixSpan` | Class | `(...)` |
| `mllib.fpm.PrefixSpanModel` | Class | `(...)` |
| `mllib.fpm.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.fpm.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.fpm.T` | Object | `` |
| `mllib.fpm.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.fpm.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.fpm.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.linalg.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `mllib.linalg.BooleanType` | Class | `(...)` |
| `mllib.linalg.ByteType` | Class | `(...)` |
| `mllib.linalg.DenseMatrix` | Class | `(numRows: int, numCols: int, values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `mllib.linalg.DenseVector` | Class | `(ar: Union[bytes, np.ndarray, Iterable[float]])` |
| `mllib.linalg.DoubleType` | Class | `(...)` |
| `mllib.linalg.IntegerType` | Class | `(...)` |
| `mllib.linalg.Matrices` | Class | `(...)` |
| `mllib.linalg.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `mllib.linalg.MatrixUDT` | Class | `(...)` |
| `mllib.linalg.NormType` | Object | `` |
| `mllib.linalg.QRDecomposition` | Class | `(Q: QT, R: RT)` |
| `mllib.linalg.QT` | Object | `` |
| `mllib.linalg.RT` | Object | `` |
| `mllib.linalg.SparseMatrix` | Class | `(numRows: int, numCols: int, colPtrs: Union[bytes, Iterable[int]], rowIndices: Union[bytes, Iterable[int]], values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `mllib.linalg.SparseVector` | Class | `(size: int, args: Union[bytes, Tuple[int, float], Iterable[float], Iterable[Tuple[int, float]], Dict[int, float]])` |
| `mllib.linalg.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `mllib.linalg.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `mllib.linalg.UserDefinedType` | Class | `(...)` |
| `mllib.linalg.Vector` | Class | `(...)` |
| `mllib.linalg.VectorLike` | Object | `` |
| `mllib.linalg.VectorUDT` | Class | `(...)` |
| `mllib.linalg.Vectors` | Class | `(...)` |
| `mllib.linalg.distributed.BlockMatrix` | Class | `(blocks: RDD[Tuple[Tuple[int, int], Matrix]], rowsPerBlock: int, colsPerBlock: int, numRows: int, numCols: int)` |
| `mllib.linalg.distributed.CoordinateMatrix` | Class | `(entries: RDD[Union[Tuple[int, int, float], MatrixEntry]], numRows: int, numCols: int)` |
| `mllib.linalg.distributed.DataFrame` | Class | `(...)` |
| `mllib.linalg.distributed.DenseMatrix` | Class | `(numRows: int, numCols: int, values: Union[bytes, Iterable[float]], isTransposed: bool)` |
| `mllib.linalg.distributed.DistributedMatrix` | Class | `(...)` |
| `mllib.linalg.distributed.IndexedRow` | Class | `(index: int, vector: VectorLike)` |
| `mllib.linalg.distributed.IndexedRowMatrix` | Class | `(rows: RDD[Union[Tuple[int, VectorLike], IndexedRow]], numRows: int, numCols: int)` |
| `mllib.linalg.distributed.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.linalg.distributed.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `mllib.linalg.distributed.MatrixEntry` | Class | `(i: int, j: int, value: float)` |
| `mllib.linalg.distributed.MultivariateStatisticalSummary` | Class | `(...)` |
| `mllib.linalg.distributed.QRDecomposition` | Class | `(Q: QT, R: RT)` |
| `mllib.linalg.distributed.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.linalg.distributed.RowMatrix` | Class | `(rows: Union[RDD[Vector], DataFrame], numRows: int, numCols: int)` |
| `mllib.linalg.distributed.SingularValueDecomposition` | Class | `(...)` |
| `mllib.linalg.distributed.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `mllib.linalg.distributed.UT` | Object | `` |
| `mllib.linalg.distributed.VT` | Object | `` |
| `mllib.linalg.distributed.Vector` | Class | `(...)` |
| `mllib.linalg.distributed.VectorLike` | Object | `` |
| `mllib.linalg.distributed.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.linalg.distributed.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.linalg.newlinalg` | Object | `` |
| `mllib.linalg.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.random.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.random.RandomRDDs` | Class | `(...)` |
| `mllib.random.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.random.Vector` | Class | `(...)` |
| `mllib.random.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.random.toArray` | Function | `(f: Callable[..., RDD[Vector]]) -> Callable[..., RDD[np.ndarray]]` |
| `mllib.recommendation.ALS` | Class | `(...)` |
| `mllib.recommendation.DataFrame` | Class | `(...)` |
| `mllib.recommendation.JavaLoader` | Class | `(...)` |
| `mllib.recommendation.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.recommendation.JavaSaveable` | Class | `(...)` |
| `mllib.recommendation.MatrixFactorizationModel` | Class | `(...)` |
| `mllib.recommendation.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.recommendation.Rating` | Class | `(...)` |
| `mllib.recommendation.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.recommendation.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.recommendation.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.recommendation.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.regression.DStream` | Class | `(jdstream: JavaObject, ssc: StreamingContext, jrdd_deserializer: Serializer)` |
| `mllib.regression.IsotonicRegression` | Class | `(...)` |
| `mllib.regression.IsotonicRegressionModel` | Class | `(boundaries: np.ndarray, predictions: np.ndarray, isotonic: bool)` |
| `mllib.regression.K` | Object | `` |
| `mllib.regression.LM` | Object | `` |
| `mllib.regression.LabeledPoint` | Class | `(label: float, features: Iterable[float])` |
| `mllib.regression.LassoModel` | Class | `(...)` |
| `mllib.regression.LassoWithSGD` | Class | `(...)` |
| `mllib.regression.LinearModel` | Class | `(weights: Vector, intercept: float)` |
| `mllib.regression.LinearRegressionModel` | Class | `(...)` |
| `mllib.regression.LinearRegressionModelBase` | Class | `(...)` |
| `mllib.regression.LinearRegressionWithSGD` | Class | `(...)` |
| `mllib.regression.Loader` | Class | `(...)` |
| `mllib.regression.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.regression.RidgeRegressionModel` | Class | `(...)` |
| `mllib.regression.RidgeRegressionWithSGD` | Class | `(...)` |
| `mllib.regression.Saveable` | Class | `(...)` |
| `mllib.regression.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.regression.StreamingLinearAlgorithm` | Class | `(model: Optional[LinearModel])` |
| `mllib.regression.StreamingLinearRegressionWithSGD` | Class | `(stepSize: float, numIterations: int, miniBatchFraction: float, convergenceTol: float)` |
| `mllib.regression.Vector` | Class | `(...)` |
| `mllib.regression.VectorLike` | Object | `` |
| `mllib.regression.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.regression.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.regression.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.stat.ChiSqTestResult` | Class | `(...)` |
| `mllib.stat.KernelDensity.KernelDensity` | Class | `()` |
| `mllib.stat.KernelDensity.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.stat.KernelDensity.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.stat.KolmogorovSmirnovTestResult` | Class | `(...)` |
| `mllib.stat.MultivariateGaussian` | Class | `(...)` |
| `mllib.stat.MultivariateStatisticalSummary` | Class | `(...)` |
| `mllib.stat.Statistics` | Class | `(...)` |
| `mllib.stat.distribution.Matrix` | Class | `(numRows: int, numCols: int, isTransposed: bool)` |
| `mllib.stat.distribution.MultivariateGaussian` | Class | `(...)` |
| `mllib.stat.distribution.Vector` | Class | `(...)` |
| `mllib.stat.test.ChiSqTestResult` | Class | `(...)` |
| `mllib.stat.test.DF` | Object | `` |
| `mllib.stat.test.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.stat.test.KolmogorovSmirnovTestResult` | Class | `(...)` |
| `mllib.stat.test.TestResult` | Class | `(...)` |
| `mllib.stat.test.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.tree.DecisionTree` | Class | `(...)` |
| `mllib.tree.DecisionTreeModel` | Class | `(...)` |
| `mllib.tree.GradientBoostedTrees` | Class | `(...)` |
| `mllib.tree.GradientBoostedTreesModel` | Class | `(...)` |
| `mllib.tree.JavaLoader` | Class | `(...)` |
| `mllib.tree.JavaModelWrapper` | Class | `(java_model: JavaObject)` |
| `mllib.tree.JavaSaveable` | Class | `(...)` |
| `mllib.tree.LabeledPoint` | Class | `(label: float, features: Iterable[float])` |
| `mllib.tree.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.tree.RandomForest` | Class | `(...)` |
| `mllib.tree.RandomForestModel` | Class | `(...)` |
| `mllib.tree.TreeEnsembleModel` | Class | `(...)` |
| `mllib.tree.VectorLike` | Object | `` |
| `mllib.tree.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.tree.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.tree.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.util.DataFrame` | Class | `(...)` |
| `mllib.util.JL` | Object | `` |
| `mllib.util.JavaLoader` | Class | `(...)` |
| `mllib.util.JavaSaveable` | Class | `(...)` |
| `mllib.util.L` | Object | `` |
| `mllib.util.LabeledPoint` | Class | `(label: float, features: Iterable[float])` |
| `mllib.util.LinearDataGenerator` | Class | `(...)` |
| `mllib.util.Loader` | Class | `(...)` |
| `mllib.util.MLUtils` | Class | `(...)` |
| `mllib.util.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `mllib.util.Saveable` | Class | `(...)` |
| `mllib.util.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `mllib.util.SparseVector` | Class | `(size: int, args: Union[bytes, Tuple[int, float], Iterable[float], Iterable[Tuple[int, float]], Dict[int, float]])` |
| `mllib.util.T` | Object | `` |
| `mllib.util.Vector` | Class | `(...)` |
| `mllib.util.VectorLike` | Object | `` |
| `mllib.util.Vectors` | Class | `(...)` |
| `mllib.util.callMLlibFunc` | Function | `(name: str, args: Any) -> Any` |
| `mllib.util.inherit_doc` | Function | `(cls: C) -> C` |
| `mllib.util.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `mllib.ver` | Object | `` |
| `pandas.CategoricalIndex` | Class | `(...)` |
| `pandas.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.DatetimeIndex` | Class | `(...)` |
| `pandas.Index` | Class | `(...)` |
| `pandas.MissingPandasLikeGeneralFunctions` | Class | `(...)` |
| `pandas.MissingPandasLikeScalars` | Class | `(...)` |
| `pandas.MultiIndex` | Class | `(...)` |
| `pandas.NamedAgg` | Object | `` |
| `pandas.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.TimedeltaIndex` | Class | `(...)` |
| `pandas.accessors.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.accessors.DataFrameOrSeries` | Object | `` |
| `pandas.accessors.DataFrameType` | Class | `(index_fields: List[InternalField], data_fields: List[InternalField])` |
| `pandas.accessors.DataType` | Class | `(...)` |
| `pandas.accessors.F` | Object | `` |
| `pandas.accessors.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.accessors.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.accessors.LongType` | Class | `(...)` |
| `pandas.accessors.Name` | Object | `` |
| `pandas.accessors.PandasOnSparkFrameMethods` | Class | `(frame: DataFrame)` |
| `pandas.accessors.PandasOnSparkSeriesMethods` | Class | `(series: Series)` |
| `pandas.accessors.SPARK_DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.accessors.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.accessors.SPARK_INDEX_NAME_PATTERN` | Object | `` |
| `pandas.accessors.ScalarType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.accessors.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.accessors.SeriesType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.accessors.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.accessors.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.accessors.UserDefinedFunctionLike` | Class | `(...)` |
| `pandas.accessors.infer_return_type` | Function | `(f: Callable) -> Union[SeriesType, DataFrameType, ScalarType, UnknownType]` |
| `pandas.accessors.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.accessors.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.accessors.log_advice` | Function | `(message: str) -> None` |
| `pandas.accessors.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.accessors.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `pandas.accessors.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.accessors.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.base.Axis` | Object | `` |
| `pandas.base.BooleanType` | Class | `(...)` |
| `pandas.base.Column` | Class | `(...)` |
| `pandas.base.ColumnOrName` | Object | `` |
| `pandas.base.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.base.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.base.Dtype` | Object | `` |
| `pandas.base.ERROR_MESSAGE_CANNOT_COMBINE` | Object | `` |
| `pandas.base.F` | Object | `` |
| `pandas.base.IndexOpsLike` | Object | `` |
| `pandas.base.IndexOpsMixin` | Class | `(...)` |
| `pandas.base.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.base.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.base.Label` | Object | `` |
| `pandas.base.LongType` | Class | `(...)` |
| `pandas.base.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.base.NumericType` | Class | `(...)` |
| `pandas.base.SPARK_DEFAULT_INDEX_NAME` | Object | `` |
| `pandas.base.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.base.SeriesOrIndex` | Object | `` |
| `pandas.base.SparkIndexOpsMethods` | Class | `(data: IndexOpsLike)` |
| `pandas.base.Window` | Class | `(...)` |
| `pandas.base.align_diff_index_ops` | Function | `(func: Callable[..., Column], this_index_ops: SeriesOrIndex, args: Any) -> SeriesOrIndex` |
| `pandas.base.ansi_mode_context` | Function | `(spark: SparkSession) -> Iterator[None]` |
| `pandas.base.booleanize_null` | Function | `(scol: Column, f: Callable[..., Column]) -> Column` |
| `pandas.base.column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.base.combine_frames` | Function | `(this: DataFrame, args: DataFrameOrSeries, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.base.extension_dtypes` | Object | `` |
| `pandas.base.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.base.numpy_column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.base.option_context` | Function | `(args: Any) -> Iterator[None]` |
| `pandas.base.ps` | Object | `` |
| `pandas.base.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.base.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.base.should_alignment_for_column_op` | Function | `(self: SeriesOrIndex, other: SeriesOrIndex) -> bool` |
| `pandas.base.validate_axis` | Function | `(axis: Optional[Axis], none_axis: int) -> int` |
| `pandas.broadcast` | Function | `(obj: DataFrame) -> DataFrame` |
| `pandas.categorical.CategoricalAccessor` | Class | `(series: ps.Series)` |
| `pandas.categorical.F` | Object | `` |
| `pandas.categorical.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.categorical.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.categorical.ps` | Object | `` |
| `pandas.concat` | Function | `(objs: List[Union[DataFrame, Series]], axis: Axis, join: str, ignore_index: bool, sort: bool) -> Union[Series, DataFrame]` |
| `pandas.config.DictWrapper` | Class | `(d: Dict[str, Option], prefix: str)` |
| `pandas.config.Option` | Class | `(key: str, doc: str, default: Any, types: Union[Tuple[type, ...], type], check_func: Tuple[Callable[[Any], bool], str])` |
| `pandas.config.OptionError` | Class | `(...)` |
| `pandas.config.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pandas.config.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.config.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.config.option_context` | Function | `(args: Any) -> Iterator[None]` |
| `pandas.config.options` | Object | `` |
| `pandas.config.reset_option` | Function | `(key: str, spark_session: Optional[SparkSession]) -> None` |
| `pandas.config.set_option` | Function | `(key: str, value: Any, spark_session: Optional[SparkSession]) -> None` |
| `pandas.config.show_options` | Function | `() -> None` |
| `pandas.correlation.CORRELATION_CORR_OUTPUT_COLUMN` | Object | `` |
| `pandas.correlation.CORRELATION_COUNT_OUTPUT_COLUMN` | Object | `` |
| `pandas.correlation.CORRELATION_VALUE_1_COLUMN` | Object | `` |
| `pandas.correlation.CORRELATION_VALUE_2_COLUMN` | Object | `` |
| `pandas.correlation.F` | Object | `` |
| `pandas.correlation.SparkDataFrame` | Class | `(...)` |
| `pandas.correlation.Window` | Class | `(...)` |
| `pandas.correlation.compute` | Function | `(sdf: SparkDataFrame, groupKeys: List[str], method: str) -> SparkDataFrame` |
| `pandas.correlation.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.correlation.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.data_type_ops.base.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `pandas.data_type_ops.base.BinaryType` | Class | `(...)` |
| `pandas.data_type_ops.base.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.base.DataType` | Class | `(...)` |
| `pandas.data_type_ops.base.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.base.DateType` | Class | `(...)` |
| `pandas.data_type_ops.base.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `pandas.data_type_ops.base.DecimalType` | Class | `(precision: int, scale: int)` |
| `pandas.data_type_ops.base.Dtype` | Object | `` |
| `pandas.data_type_ops.base.F` | Object | `` |
| `pandas.data_type_ops.base.FractionalType` | Class | `(...)` |
| `pandas.data_type_ops.base.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.base.IntegralType` | Class | `(...)` |
| `pandas.data_type_ops.base.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `pandas.data_type_ops.base.NullType` | Class | `(...)` |
| `pandas.data_type_ops.base.NumericType` | Class | `(...)` |
| `pandas.data_type_ops.base.PySparkColumn` | Class | `(...)` |
| `pandas.data_type_ops.base.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.base.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.base.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.data_type_ops.base.TimestampNTZType` | Class | `(...)` |
| `pandas.data_type_ops.base.TimestampType` | Class | `(...)` |
| `pandas.data_type_ops.base.UserDefinedType` | Class | `(...)` |
| `pandas.data_type_ops.base.extension_dtypes` | Object | `` |
| `pandas.data_type_ops.base.extension_dtypes_available` | Object | `` |
| `pandas.data_type_ops.base.extension_float_dtypes_available` | Object | `` |
| `pandas.data_type_ops.base.extension_object_dtypes_available` | Object | `` |
| `pandas.data_type_ops.base.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.data_type_ops.base.is_valid_operand_for_numeric_arithmetic` | Function | `(operand: Any, allow_bool: bool) -> bool` |
| `pandas.data_type_ops.base.spark_type_to_pandas_dtype` | Function | `(spark_type: types.DataType, use_extension_dtypes: bool) -> Dtype` |
| `pandas.data_type_ops.base.transform_boolean_operand_to_numeric` | Function | `(operand: Any, spark_type: Optional[DataType]) -> Any` |
| `pandas.data_type_ops.binary_ops.BinaryOps` | Class | `(...)` |
| `pandas.data_type_ops.binary_ops.BinaryType` | Class | `(...)` |
| `pandas.data_type_ops.binary_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.binary_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.binary_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.binary_ops.F` | Object | `` |
| `pandas.data_type_ops.binary_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.binary_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.binary_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.binary_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.binary_ops.column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.data_type_ops.binary_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.binary_ops.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `pandas.data_type_ops.boolean_ops.BooleanExtensionOps` | Class | `(...)` |
| `pandas.data_type_ops.boolean_ops.BooleanOps` | Class | `(...)` |
| `pandas.data_type_ops.boolean_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.boolean_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.boolean_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.boolean_ops.F` | Object | `` |
| `pandas.data_type_ops.boolean_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.boolean_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.boolean_ops.PySparkColumn` | Class | `(...)` |
| `pandas.data_type_ops.boolean_ops.PySparkValueError` | Class | `(...)` |
| `pandas.data_type_ops.boolean_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.boolean_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.boolean_ops.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.data_type_ops.boolean_ops.column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.data_type_ops.boolean_ops.extension_dtypes` | Object | `` |
| `pandas.data_type_ops.boolean_ops.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.data_type_ops.boolean_ops.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.data_type_ops.boolean_ops.is_valid_operand_for_numeric_arithmetic` | Function | `(operand: Any, allow_bool: bool) -> bool` |
| `pandas.data_type_ops.boolean_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.boolean_ops.transform_boolean_operand_to_numeric` | Function | `(operand: Any, spark_type: Optional[DataType]) -> Any` |
| `pandas.data_type_ops.categorical_ops.CategoricalOps` | Class | `(...)` |
| `pandas.data_type_ops.categorical_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.categorical_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.categorical_ops.F` | Object | `` |
| `pandas.data_type_ops.categorical_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.categorical_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.categorical_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.categorical_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.categorical_ops.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `pandas.data_type_ops.complex_ops.ArrayOps` | Class | `(...)` |
| `pandas.data_type_ops.complex_ops.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `pandas.data_type_ops.complex_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.complex_ops.Column` | Class | `(...)` |
| `pandas.data_type_ops.complex_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.complex_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.complex_ops.F` | Object | `` |
| `pandas.data_type_ops.complex_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.complex_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.complex_ops.MapOps` | Class | `(...)` |
| `pandas.data_type_ops.complex_ops.NumericType` | Class | `(...)` |
| `pandas.data_type_ops.complex_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.complex_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.complex_ops.StructOps` | Class | `(...)` |
| `pandas.data_type_ops.complex_ops.column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.data_type_ops.complex_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.date_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.date_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.date_ops.DateOps` | Class | `(...)` |
| `pandas.data_type_ops.date_ops.DateType` | Class | `(...)` |
| `pandas.data_type_ops.date_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.date_ops.F` | Object | `` |
| `pandas.data_type_ops.date_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.date_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.date_ops.PySparkColumn` | Class | `(...)` |
| `pandas.data_type_ops.date_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.date_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.date_ops.column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.data_type_ops.date_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.datetime_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.Column` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.datetime_ops.DatetimeNTZOps` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.DatetimeOps` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.datetime_ops.F` | Object | `` |
| `pandas.data_type_ops.datetime_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.datetime_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.LongType` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.NumericType` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.SF` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.datetime_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.datetime_ops.TimestampNTZType` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.TimestampType` | Class | `(...)` |
| `pandas.data_type_ops.datetime_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.datetime_ops.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `pandas.data_type_ops.null_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.null_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.null_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.null_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.null_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.null_ops.NullOps` | Class | `(...)` |
| `pandas.data_type_ops.null_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.null_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.null_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.null_ops.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `pandas.data_type_ops.num_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.DataType` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.num_ops.DecimalOps` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.DecimalType` | Class | `(precision: int, scale: int)` |
| `pandas.data_type_ops.num_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.num_ops.F` | Object | `` |
| `pandas.data_type_ops.num_ops.FractionalExtensionOps` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.FractionalOps` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.num_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.IntegralExtensionOps` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.IntegralOps` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.NumericOps` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.PySparkColumn` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.PySparkValueError` | Class | `(...)` |
| `pandas.data_type_ops.num_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.num_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.num_ops.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.data_type_ops.num_ops.column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.data_type_ops.num_ops.extension_dtypes` | Object | `` |
| `pandas.data_type_ops.num_ops.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.data_type_ops.num_ops.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.data_type_ops.num_ops.is_valid_operand_for_numeric_arithmetic` | Function | `(operand: Any, allow_bool: bool) -> bool` |
| `pandas.data_type_ops.num_ops.numpy_column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.data_type_ops.num_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.num_ops.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `pandas.data_type_ops.num_ops.transform_boolean_operand_to_numeric` | Function | `(operand: Any, spark_type: Optional[DataType]) -> Any` |
| `pandas.data_type_ops.string_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.string_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.string_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.string_ops.F` | Object | `` |
| `pandas.data_type_ops.string_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.string_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.string_ops.IntegralType` | Class | `(...)` |
| `pandas.data_type_ops.string_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.string_ops.StringExtensionOps` | Class | `(...)` |
| `pandas.data_type_ops.string_ops.StringOps` | Class | `(...)` |
| `pandas.data_type_ops.string_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.string_ops.column_op` | Function | `(f: Callable[..., Column]) -> Callable[..., SeriesOrIndex]` |
| `pandas.data_type_ops.string_ops.extension_dtypes` | Object | `` |
| `pandas.data_type_ops.string_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.string_ops.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `pandas.data_type_ops.timedelta_ops.BooleanType` | Class | `(...)` |
| `pandas.data_type_ops.timedelta_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.timedelta_ops.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `pandas.data_type_ops.timedelta_ops.Dtype` | Object | `` |
| `pandas.data_type_ops.timedelta_ops.IndexOpsLike` | Object | `` |
| `pandas.data_type_ops.timedelta_ops.IndexOpsMixin` | Class | `(...)` |
| `pandas.data_type_ops.timedelta_ops.SeriesOrIndex` | Object | `` |
| `pandas.data_type_ops.timedelta_ops.StringType` | Class | `(collation: str)` |
| `pandas.data_type_ops.timedelta_ops.TimedeltaOps` | Class | `(...)` |
| `pandas.data_type_ops.timedelta_ops.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.data_type_ops.timedelta_ops.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `pandas.data_type_ops.udt_ops.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.data_type_ops.udt_ops.UDTOps` | Class | `(...)` |
| `pandas.date_range` | Function | `(start: Union[str, Any], end: Union[str, Any], periods: Optional[int], freq: Optional[Union[str, DateOffset]], tz: Optional[Union[str, tzinfo]], normalize: bool, name: Optional[str], inclusive: str, kwargs: Any) -> DatetimeIndex` |
| `pandas.datetimes.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.datetimes.DateType` | Class | `(...)` |
| `pandas.datetimes.DatetimeMethods` | Class | `(series: ps.Series)` |
| `pandas.datetimes.F` | Object | `` |
| `pandas.datetimes.IntegerType` | Class | `(...)` |
| `pandas.datetimes.TimestampNTZType` | Class | `(...)` |
| `pandas.datetimes.TimestampType` | Class | `(...)` |
| `pandas.datetimes.option_context` | Function | `(args: Any) -> Iterator[None]` |
| `pandas.datetimes.ps` | Object | `` |
| `pandas.exceptions.DataError` | Class | `(...)` |
| `pandas.exceptions.PandasNotImplementedError` | Class | `(class_name: str, method_name: Optional[str], arg_name: Optional[str], property_name: Optional[str], scalar_name: Optional[str], deprecated: bool, reason: str)` |
| `pandas.exceptions.SparkPandasIndexingError` | Class | `(...)` |
| `pandas.exceptions.SparkPandasNotImplementedError` | Class | `(pandas_function: str, spark_target_function: str, description: str)` |
| `pandas.exceptions.code_change_hint` | Function | `(pandas_function: str, spark_target_function: str) -> str` |
| `pandas.extensions.CachedAccessor` | Class | `(name: str, accessor: Type[T])` |
| `pandas.extensions.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.extensions.Index` | Class | `(...)` |
| `pandas.extensions.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.extensions.T` | Object | `` |
| `pandas.extensions.register_dataframe_accessor` | Function | `(name: str) -> Callable[[Type[T]], Type[T]]` |
| `pandas.extensions.register_index_accessor` | Function | `(name: str) -> Callable[[Type[T]], Type[T]]` |
| `pandas.extensions.register_series_accessor` | Function | `(name: str) -> Callable[[Type[T]], Type[T]]` |
| `pandas.frame.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `pandas.frame.Axis` | Object | `` |
| `pandas.frame.BooleanType` | Class | `(...)` |
| `pandas.frame.CORRELATION_CORR_OUTPUT_COLUMN` | Object | `` |
| `pandas.frame.CORRELATION_COUNT_OUTPUT_COLUMN` | Object | `` |
| `pandas.frame.CORRELATION_VALUE_1_COLUMN` | Object | `` |
| `pandas.frame.CORRELATION_VALUE_2_COLUMN` | Object | `` |
| `pandas.frame.CachedDataFrame` | Class | `(internal: InternalFrame, storage_level: Optional[StorageLevel])` |
| `pandas.frame.CachedSparkFrameMethods` | Class | `(frame: CachedDataFrame)` |
| `pandas.frame.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.frame.DataFrameGroupBy` | Class | `(psdf: DataFrame, by: List[Series], as_index: bool, dropna: bool, column_labels_to_exclude: Set[Label], agg_columns: List[Label])` |
| `pandas.frame.DataFrameOrSeries` | Object | `` |
| `pandas.frame.DataFrameResampler` | Class | `(psdf: DataFrame, resamplekey: Optional[Series], rule: str, closed: Optional[str], label: Optional[str], agg_columns: List[Series])` |
| `pandas.frame.DataFrameType` | Class | `(index_fields: List[InternalField], data_fields: List[InternalField])` |
| `pandas.frame.DataType` | Class | `(...)` |
| `pandas.frame.DecimalType` | Class | `(precision: int, scale: int)` |
| `pandas.frame.DoubleType` | Class | `(...)` |
| `pandas.frame.Dtype` | Object | `` |
| `pandas.frame.F` | Object | `` |
| `pandas.frame.Frame` | Class | `(...)` |
| `pandas.frame.FuncT` | Object | `` |
| `pandas.frame.HIDDEN_COLUMNS` | Object | `` |
| `pandas.frame.Index` | Class | `(...)` |
| `pandas.frame.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.frame.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.frame.Label` | Object | `` |
| `pandas.frame.MissingPandasLikeDataFrame` | Class | `(...)` |
| `pandas.frame.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.frame.Name` | Object | `` |
| `pandas.frame.NullType` | Class | `(...)` |
| `pandas.frame.NumericType` | Class | `(...)` |
| `pandas.frame.OptionalPrimitiveType` | Object | `` |
| `pandas.frame.PandasOnSparkFrameMethods` | Class | `(frame: DataFrame)` |
| `pandas.frame.PandasOnSparkPlotAccessor` | Class | `(data)` |
| `pandas.frame.PySparkColumn` | Class | `(...)` |
| `pandas.frame.PySparkDataFrame` | Class | `(...)` |
| `pandas.frame.PySparkValueError` | Class | `(...)` |
| `pandas.frame.REPR_HTML_PATTERN` | Object | `` |
| `pandas.frame.REPR_PATTERN` | Object | `` |
| `pandas.frame.Row` | Class | `(...)` |
| `pandas.frame.SF` | Class | `(...)` |
| `pandas.frame.SPARK_DEFAULT_INDEX_NAME` | Object | `` |
| `pandas.frame.SPARK_DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.frame.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.frame.SPARK_INDEX_NAME_PATTERN` | Object | `` |
| `pandas.frame.Scalar` | Object | `` |
| `pandas.frame.ScalarType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.frame.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.frame.SeriesType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.frame.SparkFrameMethods` | Class | `(frame: ps.DataFrame)` |
| `pandas.frame.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `pandas.frame.StringType` | Class | `(collation: str)` |
| `pandas.frame.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.frame.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.frame.T` | Object | `` |
| `pandas.frame.TimestampNTZType` | Class | `(...)` |
| `pandas.frame.TimestampType` | Class | `(...)` |
| `pandas.frame.Window` | Class | `(...)` |
| `pandas.frame.align_diff_frames` | Function | `(resolve_func: Callable[[DataFrame, List[Label], List[Label]], Iterator[Tuple[Series, Label]]], this: DataFrame, that: DataFrame, fillna: bool, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.frame.ansi_mode_context` | Function | `(spark: SparkSession) -> Iterator[None]` |
| `pandas.frame.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.frame.column_labels_level` | Function | `(column_labels: List[Label]) -> int` |
| `pandas.frame.combine_frames` | Function | `(this: DataFrame, args: DataFrameOrSeries, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.frame.compute` | Function | `(sdf: SparkDataFrame, groupKeys: List[str], method: str) -> SparkDataFrame` |
| `pandas.frame.create_tuple_for_frame_type` | Function | `(params: Any) -> object` |
| `pandas.frame.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.frame.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.frame.infer_return_type` | Function | `(f: Callable) -> Union[SeriesType, DataFrameType, ScalarType, UnknownType]` |
| `pandas.frame.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.frame.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.frame.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.frame.is_testing` | Function | `() -> bool` |
| `pandas.frame.log_advice` | Function | `(message: str) -> None` |
| `pandas.frame.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.frame.option_context` | Function | `(args: Any) -> Iterator[None]` |
| `pandas.frame.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.frame.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `pandas.frame.ps` | Object | `` |
| `pandas.frame.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.frame.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.frame.spark_type_to_pandas_dtype` | Function | `(spark_type: types.DataType, use_extension_dtypes: bool) -> Dtype` |
| `pandas.frame.validate_arguments_and_invoke_function` | Function | `(pobj: Union[pd.DataFrame, pd.Series], pandas_on_spark_func: Callable, pandas_func: Callable, input_args: Dict) -> Any` |
| `pandas.frame.validate_axis` | Function | `(axis: Optional[Axis], none_axis: int) -> int` |
| `pandas.frame.validate_bool_kwarg` | Function | `(value: Any, arg_name: str) -> Optional[bool]` |
| `pandas.frame.validate_how` | Function | `(how: str) -> str` |
| `pandas.frame.validate_mode` | Function | `(mode: str) -> str` |
| `pandas.frame.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.frame.with_ansi_mode_context` | Function | `(f: FuncT) -> FuncT` |
| `pandas.from_pandas` | Function | `(pobj: Union[pd.DataFrame, pd.Series, pd.Index]) -> Union[Series, DataFrame, Index]` |
| `pandas.generic.AtIndexer` | Class | `(...)` |
| `pandas.generic.Axis` | Object | `` |
| `pandas.generic.BooleanType` | Class | `(...)` |
| `pandas.generic.Column` | Class | `(...)` |
| `pandas.generic.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.generic.DataFrameOrSeries` | Object | `` |
| `pandas.generic.DoubleType` | Class | `(...)` |
| `pandas.generic.Dtype` | Object | `` |
| `pandas.generic.Expanding` | Class | `(psdf_or_psser: FrameLike, min_periods: int)` |
| `pandas.generic.ExponentialMoving` | Class | `(psdf_or_psser: FrameLike, com: Optional[float], span: Optional[float], halflife: Optional[float], alpha: Optional[float], min_periods: Optional[int], ignore_na: bool)` |
| `pandas.generic.F` | Object | `` |
| `pandas.generic.Frame` | Class | `(...)` |
| `pandas.generic.FrameLike` | Object | `` |
| `pandas.generic.GroupBy` | Class | `(psdf: DataFrame, groupkeys: List[Series], as_index: bool, dropna: bool, column_labels_to_exclude: Set[Label], agg_columns_selected: bool, agg_columns: List[Series])` |
| `pandas.generic.Index` | Class | `(...)` |
| `pandas.generic.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.generic.Label` | Object | `` |
| `pandas.generic.LocIndexer` | Class | `(...)` |
| `pandas.generic.LongType` | Class | `(...)` |
| `pandas.generic.Name` | Object | `` |
| `pandas.generic.NumericType` | Class | `(...)` |
| `pandas.generic.Rolling` | Class | `(psdf_or_psser: FrameLike, window: int, min_periods: Optional[int])` |
| `pandas.generic.SF` | Class | `(...)` |
| `pandas.generic.SPARK_CONF_ARROW_ENABLED` | Object | `` |
| `pandas.generic.Scalar` | Object | `` |
| `pandas.generic.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.generic.bool_type` | Object | `` |
| `pandas.generic.iAtIndexer` | Class | `(...)` |
| `pandas.generic.iLocIndexer` | Class | `(...)` |
| `pandas.generic.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.generic.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.generic.log_advice` | Function | `(message: str) -> None` |
| `pandas.generic.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.generic.ps` | Object | `` |
| `pandas.generic.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.generic.spark_type_to_pandas_dtype` | Function | `(spark_type: types.DataType, use_extension_dtypes: bool) -> Dtype` |
| `pandas.generic.sql_conf` | Function | `(pairs: Dict[str, Any], spark: Optional[SparkSession]) -> Iterator[None]` |
| `pandas.generic.validate_arguments_and_invoke_function` | Function | `(pobj: Union[pd.DataFrame, pd.Series], pandas_on_spark_func: Callable, pandas_func: Callable, input_args: Dict) -> Any` |
| `pandas.generic.validate_axis` | Function | `(axis: Optional[Axis], none_axis: int) -> int` |
| `pandas.generic.validate_mode` | Function | `(mode: str) -> str` |
| `pandas.get_dummies` | Function | `(data: Union[DataFrame, Series], prefix: Optional[Union[str, List[str], Dict[str, str]]], prefix_sep: str, dummy_na: bool, columns: Optional[Union[Name, List[Name]]], sparse: bool, drop_first: bool, dtype: Optional[Union[str, Dtype]]) -> DataFrame` |
| `pandas.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.groupby.Axis` | Object | `` |
| `pandas.groupby.BooleanType` | Class | `(...)` |
| `pandas.groupby.CORRELATION_CORR_OUTPUT_COLUMN` | Object | `` |
| `pandas.groupby.CORRELATION_COUNT_OUTPUT_COLUMN` | Object | `` |
| `pandas.groupby.CORRELATION_VALUE_1_COLUMN` | Object | `` |
| `pandas.groupby.CORRELATION_VALUE_2_COLUMN` | Object | `` |
| `pandas.groupby.Column` | Class | `(...)` |
| `pandas.groupby.DataError` | Class | `(...)` |
| `pandas.groupby.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.groupby.DataFrameGroupBy` | Class | `(psdf: DataFrame, by: List[Series], as_index: bool, dropna: bool, column_labels_to_exclude: Set[Label], agg_columns: List[Label])` |
| `pandas.groupby.DataFrameType` | Class | `(index_fields: List[InternalField], data_fields: List[InternalField])` |
| `pandas.groupby.DataType` | Class | `(...)` |
| `pandas.groupby.DoubleType` | Class | `(...)` |
| `pandas.groupby.ExpandingGroupby` | Class | `(groupby: GroupBy[FrameLike], min_periods: int)` |
| `pandas.groupby.ExponentialMovingGroupby` | Class | `(groupby: GroupBy[FrameLike], com: Optional[float], span: Optional[float], halflife: Optional[float], alpha: Optional[float], min_periods: Optional[int], ignore_na: bool)` |
| `pandas.groupby.F` | Object | `` |
| `pandas.groupby.FrameLike` | Object | `` |
| `pandas.groupby.FuncT` | Object | `` |
| `pandas.groupby.GroupBy` | Class | `(psdf: DataFrame, groupkeys: List[Series], as_index: bool, dropna: bool, column_labels_to_exclude: Set[Label], agg_columns_selected: bool, agg_columns: List[Series])` |
| `pandas.groupby.HIDDEN_COLUMNS` | Object | `` |
| `pandas.groupby.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.groupby.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.groupby.Label` | Object | `` |
| `pandas.groupby.MissingPandasLikeDataFrameGroupBy` | Class | `(...)` |
| `pandas.groupby.MissingPandasLikeSeriesGroupBy` | Class | `(...)` |
| `pandas.groupby.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.groupby.Name` | Object | `` |
| `pandas.groupby.NamedAgg` | Object | `` |
| `pandas.groupby.NumericType` | Class | `(...)` |
| `pandas.groupby.RollingGroupby` | Class | `(groupby: GroupBy[FrameLike], window: int, min_periods: Optional[int])` |
| `pandas.groupby.SF` | Class | `(...)` |
| `pandas.groupby.SPARK_DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.groupby.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.groupby.SPARK_INDEX_NAME_PATTERN` | Object | `` |
| `pandas.groupby.ScalarType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.groupby.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.groupby.SeriesGroupBy` | Class | `(psser: Series, by: List[Series], as_index: bool, dropna: bool)` |
| `pandas.groupby.SeriesType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.groupby.SparkDataFrame` | Class | `(...)` |
| `pandas.groupby.StringType` | Class | `(collation: str)` |
| `pandas.groupby.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.groupby.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.groupby.Window` | Class | `(...)` |
| `pandas.groupby.align_diff_frames` | Function | `(resolve_func: Callable[[DataFrame, List[Label], List[Label]], Iterator[Tuple[Series, Label]]], this: DataFrame, that: DataFrame, fillna: bool, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.groupby.ansi_mode_context` | Function | `(spark: SparkSession) -> Iterator[None]` |
| `pandas.groupby.as_nullable_spark_type` | Function | `(dt: DataType) -> DataType` |
| `pandas.groupby.compute` | Function | `(sdf: SparkDataFrame, groupKeys: List[str], method: str) -> SparkDataFrame` |
| `pandas.groupby.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.groupby.force_decimal_precision_scale` | Function | `(dt: DataType, precision: int, scale: int) -> DataType` |
| `pandas.groupby.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.groupby.infer_return_type` | Function | `(f: Callable) -> Union[SeriesType, DataFrameType, ScalarType, UnknownType]` |
| `pandas.groupby.is_multi_agg_with_relabel` | Function | `(kwargs: Any) -> bool` |
| `pandas.groupby.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.groupby.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.groupby.log_advice` | Function | `(message: str) -> None` |
| `pandas.groupby.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.groupby.normalize_keyword_aggregation` | Function | `(kwargs: Dict[str, Tuple[Name, str]]) -> Tuple[Dict[Name, List[str]], List[str], List[Tuple]]` |
| `pandas.groupby.ps` | Object | `` |
| `pandas.groupby.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.groupby.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.groupby.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.groupby.with_ansi_mode_context` | Function | `(f: FuncT) -> FuncT` |
| `pandas.indexes.DatetimeIndex` | Class | `(...)` |
| `pandas.indexes.Index` | Class | `(...)` |
| `pandas.indexes.MultiIndex` | Class | `(...)` |
| `pandas.indexes.TimedeltaIndex` | Class | `(...)` |
| `pandas.indexes.base.Column` | Class | `(...)` |
| `pandas.indexes.base.DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.indexes.base.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.indexes.base.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `pandas.indexes.base.Dtype` | Object | `` |
| `pandas.indexes.base.ERROR_MESSAGE_CANNOT_COMBINE` | Object | `` |
| `pandas.indexes.base.F` | Object | `` |
| `pandas.indexes.base.Index` | Class | `(...)` |
| `pandas.indexes.base.IndexOpsMixin` | Class | `(...)` |
| `pandas.indexes.base.IntegralType` | Class | `(...)` |
| `pandas.indexes.base.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.indexes.base.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.indexes.base.Label` | Object | `` |
| `pandas.indexes.base.MissingPandasLikeIndex` | Class | `(...)` |
| `pandas.indexes.base.Name` | Object | `` |
| `pandas.indexes.base.SPARK_DEFAULT_INDEX_NAME` | Object | `` |
| `pandas.indexes.base.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.indexes.base.Scalar` | Object | `` |
| `pandas.indexes.base.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.indexes.base.SparkIndexMethods` | Class | `(...)` |
| `pandas.indexes.base.SparkIndexOpsMethods` | Class | `(data: IndexOpsLike)` |
| `pandas.indexes.base.TimestampNTZType` | Class | `(...)` |
| `pandas.indexes.base.TimestampType` | Class | `(...)` |
| `pandas.indexes.base.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.indexes.base.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.indexes.base.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.indexes.base.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.indexes.base.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.indexes.base.log_advice` | Function | `(message: str) -> None` |
| `pandas.indexes.base.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.indexes.base.option_context` | Function | `(args: Any) -> Iterator[None]` |
| `pandas.indexes.base.ps` | Object | `` |
| `pandas.indexes.base.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.indexes.base.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.indexes.base.validate_bool_kwarg` | Function | `(value: Any, arg_name: str) -> Optional[bool]` |
| `pandas.indexes.base.validate_index_loc` | Function | `(index: Index, loc: int) -> None` |
| `pandas.indexes.base.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.indexes.base.xor` | Function | `(df1: PySparkDataFrame, df2: PySparkDataFrame) -> PySparkDataFrame` |
| `pandas.indexes.category.CategoricalIndex` | Class | `(...)` |
| `pandas.indexes.category.Index` | Class | `(...)` |
| `pandas.indexes.category.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.indexes.category.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.indexes.category.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.indexes.category.ps` | Object | `` |
| `pandas.indexes.datetimes.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.indexes.datetimes.DatetimeIndex` | Class | `(...)` |
| `pandas.indexes.datetimes.Index` | Class | `(...)` |
| `pandas.indexes.datetimes.MissingPandasLikeDatetimeIndex` | Class | `(...)` |
| `pandas.indexes.datetimes.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.indexes.datetimes.disallow_nanoseconds` | Function | `(freq: Union[str, DateOffset]) -> None` |
| `pandas.indexes.datetimes.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.indexes.datetimes.ps` | Object | `` |
| `pandas.indexes.datetimes.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.indexes.multi.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.indexes.multi.DataType` | Class | `(...)` |
| `pandas.indexes.multi.F` | Object | `` |
| `pandas.indexes.multi.Index` | Class | `(...)` |
| `pandas.indexes.multi.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.indexes.multi.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.indexes.multi.Label` | Object | `` |
| `pandas.indexes.multi.MissingPandasLikeMultiIndex` | Class | `(...)` |
| `pandas.indexes.multi.MultiIndex` | Class | `(...)` |
| `pandas.indexes.multi.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.indexes.multi.Name` | Object | `` |
| `pandas.indexes.multi.PandasNotImplementedError` | Class | `(class_name: str, method_name: Optional[str], arg_name: Optional[str], property_name: Optional[str], scalar_name: Optional[str], deprecated: bool, reason: str)` |
| `pandas.indexes.multi.PySparkColumn` | Class | `(...)` |
| `pandas.indexes.multi.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.indexes.multi.Scalar` | Object | `` |
| `pandas.indexes.multi.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.indexes.multi.Window` | Class | `(...)` |
| `pandas.indexes.multi.compare_disallow_null` | Function | `(left: Column, right: Column, comp: Callable[[Column, Column], Column]) -> Column` |
| `pandas.indexes.multi.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.indexes.multi.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.indexes.multi.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.indexes.multi.ps` | Object | `` |
| `pandas.indexes.multi.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.indexes.multi.validate_index_loc` | Function | `(index: Index, loc: int) -> None` |
| `pandas.indexes.multi.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.indexes.multi.xor` | Function | `(df1: PySparkDataFrame, df2: PySparkDataFrame) -> PySparkDataFrame` |
| `pandas.indexes.timedelta.F` | Object | `` |
| `pandas.indexes.timedelta.HOURS_PER_DAY` | Object | `` |
| `pandas.indexes.timedelta.Index` | Class | `(...)` |
| `pandas.indexes.timedelta.MICROS_PER_MILLIS` | Object | `` |
| `pandas.indexes.timedelta.MICROS_PER_SECOND` | Object | `` |
| `pandas.indexes.timedelta.MILLIS_PER_SECOND` | Object | `` |
| `pandas.indexes.timedelta.MINUTES_PER_HOUR` | Object | `` |
| `pandas.indexes.timedelta.MissingPandasLikeTimedeltaIndex` | Class | `(...)` |
| `pandas.indexes.timedelta.SECONDS_PER_DAY` | Object | `` |
| `pandas.indexes.timedelta.SECONDS_PER_HOUR` | Object | `` |
| `pandas.indexes.timedelta.SECONDS_PER_MINUTE` | Object | `` |
| `pandas.indexes.timedelta.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.indexes.timedelta.TimedeltaIndex` | Class | `(...)` |
| `pandas.indexes.timedelta.ps` | Object | `` |
| `pandas.indexing.AnalysisException` | Class | `(...)` |
| `pandas.indexing.AtIndexer` | Class | `(...)` |
| `pandas.indexing.BooleanType` | Class | `(...)` |
| `pandas.indexing.DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.indexing.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.indexing.DataType` | Class | `(...)` |
| `pandas.indexing.F` | Object | `` |
| `pandas.indexing.Frame` | Class | `(...)` |
| `pandas.indexing.IndexerLike` | Class | `(psdf_or_psser: Frame)` |
| `pandas.indexing.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.indexing.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.indexing.Label` | Object | `` |
| `pandas.indexing.LocIndexer` | Class | `(...)` |
| `pandas.indexing.LocIndexerLike` | Class | `(...)` |
| `pandas.indexing.LongType` | Class | `(...)` |
| `pandas.indexing.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.indexing.Name` | Object | `` |
| `pandas.indexing.PySparkColumn` | Class | `(...)` |
| `pandas.indexing.SPARK_DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.indexing.Scalar` | Object | `` |
| `pandas.indexing.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.indexing.SparkPandasIndexingError` | Class | `(...)` |
| `pandas.indexing.SparkPandasNotImplementedError` | Class | `(pandas_function: str, spark_target_function: str, description: str)` |
| `pandas.indexing.iAtIndexer` | Class | `(...)` |
| `pandas.indexing.iLocIndexer` | Class | `(...)` |
| `pandas.indexing.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.indexing.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.indexing.is_remote` | Function | `() -> bool` |
| `pandas.indexing.lazy_property` | Function | `(fn: Callable[[Any], Any]) -> property` |
| `pandas.indexing.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.indexing.ps` | Object | `` |
| `pandas.indexing.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.indexing.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.indexing.spark_column_equals` | Function | `(left: Column, right: Column) -> bool` |
| `pandas.indexing.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.internal.BooleanType` | Class | `(...)` |
| `pandas.internal.DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.internal.DataType` | Class | `(...)` |
| `pandas.internal.DataTypeOps` | Class | `(dtype: Dtype, spark_type: DataType)` |
| `pandas.internal.Dtype` | Object | `` |
| `pandas.internal.F` | Object | `` |
| `pandas.internal.HIDDEN_COLUMNS` | Object | `` |
| `pandas.internal.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.internal.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.internal.Label` | Object | `` |
| `pandas.internal.LongType` | Class | `(...)` |
| `pandas.internal.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.internal.PySparkColumn` | Class | `(...)` |
| `pandas.internal.PySparkDataFrame` | Class | `(...)` |
| `pandas.internal.SF` | Class | `(...)` |
| `pandas.internal.SPARK_DEFAULT_INDEX_NAME` | Object | `` |
| `pandas.internal.SPARK_DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.internal.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.internal.SPARK_INDEX_NAME_PATTERN` | Object | `` |
| `pandas.internal.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.internal.StringType` | Class | `(collation: str)` |
| `pandas.internal.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.internal.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.internal.Window` | Class | `(...)` |
| `pandas.internal.as_nullable_spark_type` | Function | `(dt: DataType) -> DataType` |
| `pandas.internal.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.internal.column_labels_level` | Function | `(column_labels: List[Label]) -> int` |
| `pandas.internal.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.internal.extension_dtypes` | Object | `` |
| `pandas.internal.force_decimal_precision_scale` | Function | `(dt: DataType, precision: int, scale: int) -> DataType` |
| `pandas.internal.infer_pd_series_spark_type` | Function | `(pser: pd.Series, dtype: Dtype, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.internal.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.internal.is_remote` | Function | `() -> bool` |
| `pandas.internal.is_testing` | Function | `() -> bool` |
| `pandas.internal.is_timestamp_ntz_preferred` | Function | `() -> bool` |
| `pandas.internal.lazy_property` | Function | `(fn: Callable[[Any], Any]) -> property` |
| `pandas.internal.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.internal.ps` | Object | `` |
| `pandas.internal.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.internal.spark_column_equals` | Function | `(left: Column, right: Column) -> bool` |
| `pandas.internal.spark_type_to_pandas_dtype` | Function | `(spark_type: types.DataType, use_extension_dtypes: bool) -> Dtype` |
| `pandas.isna` | Function | `(obj)` |
| `pandas.isnull` | Object | `` |
| `pandas.json_normalize` | Function | `(data: Union[Dict, List[Dict]], sep: str) -> DataFrame` |
| `pandas.melt` | Function | `(frame: DataFrame, id_vars: Optional[Union[Name, List[Name]]], value_vars: Optional[Union[Name, List[Name]]], var_name: Optional[Union[str, List[str]]], value_name: str) -> DataFrame` |
| `pandas.merge` | Function | `(obj: DataFrame, right: DataFrame, how: str, on: Optional[Union[Name, List[Name]]], left_on: Optional[Union[Name, List[Name]]], right_on: Optional[Union[Name, List[Name]]], left_index: bool, right_index: bool, suffixes: Tuple[str, str]) -> DataFrame` |
| `pandas.merge_asof` | Function | `(left: Union[DataFrame, Series], right: Union[DataFrame, Series], on: Optional[Name], left_on: Optional[Name], right_on: Optional[Name], left_index: bool, right_index: bool, by: Optional[Union[Name, List[Name]]], left_by: Optional[Union[Name, List[Name]]], right_by: Optional[Union[Name, List[Name]]], suffixes: Tuple[str, str], tolerance: Optional[Any], allow_exact_matches: bool, direction: str) -> DataFrame` |
| `pandas.missing.PandasNotImplementedError` | Class | `(class_name: str, method_name: Optional[str], arg_name: Optional[str], property_name: Optional[str], scalar_name: Optional[str], deprecated: bool, reason: str)` |
| `pandas.missing.common.array` | Function | `(f)` |
| `pandas.missing.common.duplicated` | Function | `(f)` |
| `pandas.missing.common.memory_usage` | Function | `(f)` |
| `pandas.missing.common.to_list` | Function | `(f)` |
| `pandas.missing.common.to_pickle` | Function | `(f)` |
| `pandas.missing.common.to_xarray` | Function | `(f)` |
| `pandas.missing.common.tolist` | Function | `(f)` |
| `pandas.missing.frame.MissingPandasLikeDataFrame` | Class | `(...)` |
| `pandas.missing.frame.common` | Object | `` |
| `pandas.missing.frame.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.frame.unsupported_property` | Function | `(class_name, property_name, deprecated, reason)` |
| `pandas.missing.general_functions.MissingPandasLikeGeneralFunctions` | Class | `(...)` |
| `pandas.missing.general_functions.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.groupby.MissingPandasLikeDataFrameGroupBy` | Class | `(...)` |
| `pandas.missing.groupby.MissingPandasLikeSeriesGroupBy` | Class | `(...)` |
| `pandas.missing.groupby.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.groupby.unsupported_property` | Function | `(class_name, property_name, deprecated, reason)` |
| `pandas.missing.indexes.MissingPandasLikeDatetimeIndex` | Class | `(...)` |
| `pandas.missing.indexes.MissingPandasLikeIndex` | Class | `(...)` |
| `pandas.missing.indexes.MissingPandasLikeMultiIndex` | Class | `(...)` |
| `pandas.missing.indexes.MissingPandasLikeTimedeltaIndex` | Class | `(...)` |
| `pandas.missing.indexes.common` | Object | `` |
| `pandas.missing.indexes.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.indexes.unsupported_property` | Function | `(class_name, property_name, deprecated, reason)` |
| `pandas.missing.resample.MissingPandasLikeDataFrameResampler` | Class | `(...)` |
| `pandas.missing.resample.MissingPandasLikeSeriesResampler` | Class | `(...)` |
| `pandas.missing.resample.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.resample.unsupported_property` | Function | `(class_name, property_name, deprecated, reason)` |
| `pandas.missing.scalars.MissingPandasLikeScalars` | Class | `(...)` |
| `pandas.missing.scalars.PandasNotImplementedError` | Class | `(class_name: str, method_name: Optional[str], arg_name: Optional[str], property_name: Optional[str], scalar_name: Optional[str], deprecated: bool, reason: str)` |
| `pandas.missing.series.MissingPandasLikeSeries` | Class | `(...)` |
| `pandas.missing.series.common` | Object | `` |
| `pandas.missing.series.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.series.unsupported_property` | Function | `(class_name, property_name, deprecated, reason)` |
| `pandas.missing.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.unsupported_property` | Function | `(class_name, property_name, deprecated, reason)` |
| `pandas.missing.window.MissingPandasLikeExpanding` | Class | `(...)` |
| `pandas.missing.window.MissingPandasLikeExpandingGroupby` | Class | `(...)` |
| `pandas.missing.window.MissingPandasLikeExponentialMoving` | Class | `(...)` |
| `pandas.missing.window.MissingPandasLikeExponentialMovingGroupby` | Class | `(...)` |
| `pandas.missing.window.MissingPandasLikeRolling` | Class | `(...)` |
| `pandas.missing.window.MissingPandasLikeRollingGroupby` | Class | `(...)` |
| `pandas.missing.window.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.missing.window.unsupported_property` | Function | `(class_name, property_name, deprecated, reason)` |
| `pandas.mlflow.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.mlflow.DataType` | Class | `(...)` |
| `pandas.mlflow.Dtype` | Object | `` |
| `pandas.mlflow.Label` | Object | `` |
| `pandas.mlflow.PythonModelWrapper` | Class | `(model_uri: str, return_type_hint: Union[str, type, Dtype])` |
| `pandas.mlflow.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.mlflow.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.mlflow.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.mlflow.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.mlflow.lazy_property` | Function | `(fn: Callable[[Any], Any]) -> property` |
| `pandas.mlflow.load_model` | Function | `(model_uri: str, predict_type: Union[str, type, Dtype]) -> PythonModelWrapper` |
| `pandas.mlflow.struct` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `pandas.namespace.Axis` | Object | `` |
| `pandas.namespace.BooleanType` | Class | `(...)` |
| `pandas.namespace.ByteType` | Class | `(...)` |
| `pandas.namespace.DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.namespace.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.namespace.DataType` | Class | `(...)` |
| `pandas.namespace.DateType` | Class | `(...)` |
| `pandas.namespace.DatetimeIndex` | Class | `(...)` |
| `pandas.namespace.DecimalType` | Class | `(precision: int, scale: int)` |
| `pandas.namespace.DoubleType` | Class | `(...)` |
| `pandas.namespace.Dtype` | Object | `` |
| `pandas.namespace.F` | Object | `` |
| `pandas.namespace.FloatType` | Class | `(...)` |
| `pandas.namespace.HIDDEN_COLUMNS` | Object | `` |
| `pandas.namespace.Index` | Class | `(...)` |
| `pandas.namespace.IndexOpsMixin` | Class | `(...)` |
| `pandas.namespace.IntegerType` | Class | `(...)` |
| `pandas.namespace.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.namespace.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.namespace.Label` | Object | `` |
| `pandas.namespace.LongType` | Class | `(...)` |
| `pandas.namespace.MultiIndex` | Class | `(...)` |
| `pandas.namespace.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.namespace.Name` | Object | `` |
| `pandas.namespace.PySparkColumn` | Class | `(...)` |
| `pandas.namespace.PySparkDataFrame` | Class | `(...)` |
| `pandas.namespace.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.namespace.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.namespace.ShortType` | Class | `(...)` |
| `pandas.namespace.StringType` | Class | `(collation: str)` |
| `pandas.namespace.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.namespace.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.namespace.TimedeltaIndex` | Class | `(...)` |
| `pandas.namespace.TimestampNTZType` | Class | `(...)` |
| `pandas.namespace.TimestampType` | Class | `(...)` |
| `pandas.namespace.align_diff_frames` | Function | `(resolve_func: Callable[[DataFrame, List[Label], List[Label]], Iterator[Tuple[Series, Label]]], this: DataFrame, that: DataFrame, fillna: bool, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.namespace.as_nullable_spark_type` | Function | `(dt: DataType) -> DataType` |
| `pandas.namespace.broadcast` | Function | `(obj: DataFrame) -> DataFrame` |
| `pandas.namespace.concat` | Function | `(objs: List[Union[DataFrame, Series]], axis: Axis, join: str, ignore_index: bool, sort: bool) -> Union[Series, DataFrame]` |
| `pandas.namespace.date_range` | Function | `(start: Union[str, Any], end: Union[str, Any], periods: Optional[int], freq: Optional[Union[str, DateOffset]], tz: Optional[Union[str, tzinfo]], normalize: bool, name: Optional[str], inclusive: str, kwargs: Any) -> DatetimeIndex` |
| `pandas.namespace.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.namespace.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.namespace.force_decimal_precision_scale` | Function | `(dt: DataType, precision: int, scale: int) -> DataType` |
| `pandas.namespace.from_pandas` | Function | `(pobj: Union[pd.DataFrame, pd.Series, pd.Index]) -> Union[Series, DataFrame, Index]` |
| `pandas.namespace.get_dummies` | Function | `(data: Union[DataFrame, Series], prefix: Optional[Union[str, List[str], Dict[str, str]]], prefix_sep: str, dummy_na: bool, columns: Optional[Union[Name, List[Name]]], sparse: bool, drop_first: bool, dtype: Optional[Union[str, Dtype]]) -> DataFrame` |
| `pandas.namespace.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.namespace.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.namespace.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.namespace.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.namespace.isna` | Function | `(obj)` |
| `pandas.namespace.isnull` | Object | `` |
| `pandas.namespace.json_normalize` | Function | `(data: Union[Dict, List[Dict]], sep: str) -> DataFrame` |
| `pandas.namespace.log_advice` | Function | `(message: str) -> None` |
| `pandas.namespace.melt` | Function | `(frame: DataFrame, id_vars: Optional[Union[Name, List[Name]]], value_vars: Optional[Union[Name, List[Name]]], var_name: Optional[Union[str, List[str]]], value_name: str) -> DataFrame` |
| `pandas.namespace.merge` | Function | `(obj: DataFrame, right: DataFrame, how: str, on: Optional[Union[Name, List[Name]]], left_on: Optional[Union[Name, List[Name]]], right_on: Optional[Union[Name, List[Name]]], left_index: bool, right_index: bool, suffixes: Tuple[str, str]) -> DataFrame` |
| `pandas.namespace.merge_asof` | Function | `(left: Union[DataFrame, Series], right: Union[DataFrame, Series], on: Optional[Name], left_on: Optional[Name], right_on: Optional[Name], left_index: bool, right_index: bool, by: Optional[Union[Name, List[Name]]], left_by: Optional[Union[Name, List[Name]]], right_by: Optional[Union[Name, List[Name]]], suffixes: Tuple[str, str], tolerance: Optional[Any], allow_exact_matches: bool, direction: str) -> DataFrame` |
| `pandas.namespace.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.namespace.notna` | Function | `(obj)` |
| `pandas.namespace.notnull` | Object | `` |
| `pandas.namespace.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `pandas.namespace.ps` | Object | `` |
| `pandas.namespace.range` | Function | `(start: int, end: Optional[int], step: int, num_partitions: Optional[int]) -> DataFrame` |
| `pandas.namespace.read_clipboard` | Function | `(sep: str, kwargs: Any) -> DataFrame` |
| `pandas.namespace.read_csv` | Function | `(path: Union[str, List[str]], sep: str, header: Union[str, int, None], names: Optional[Union[str, List[str]]], index_col: Optional[Union[str, List[str]]], usecols: Optional[Union[List[int], List[str], Callable[[str], bool]]], dtype: Optional[Union[str, Dtype, Dict[str, Union[str, Dtype]]]], nrows: Optional[int], parse_dates: bool, quotechar: Optional[str], escapechar: Optional[str], comment: Optional[str], encoding: Optional[str], options: Any) -> Union[DataFrame, Series]` |
| `pandas.namespace.read_delta` | Function | `(path: str, version: Optional[str], timestamp: Optional[str], index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.namespace.read_excel` | Function | `(io: Union[str, Any], sheet_name: Union[str, int, List[Union[str, int]], None], header: Union[int, List[int]], names: Optional[List], index_col: Optional[List[int]], usecols: Optional[Union[int, str, List[Union[int, str]], Callable[[str], bool]]], dtype: Optional[Dict[str, Union[str, Dtype]]], engine: Optional[str], converters: Optional[Dict], true_values: Optional[Any], false_values: Optional[Any], skiprows: Optional[Union[int, List[int]]], nrows: Optional[int], na_values: Optional[Any], keep_default_na: bool, verbose: bool, parse_dates: Union[bool, List, Dict], date_parser: Optional[Callable], thousands: Optional[str], comment: Optional[str], skipfooter: int, kwds: Any) -> Union[DataFrame, Series, Dict[str, Union[DataFrame, Series]]]` |
| `pandas.namespace.read_html` | Function | `(io: Union[str, Any], match: str, flavor: Optional[str], header: Optional[Union[int, List[int]]], index_col: Optional[Union[int, List[int]]], skiprows: Optional[Union[int, List[int], slice]], attrs: Optional[Dict[str, str]], parse_dates: bool, thousands: str, encoding: Optional[str], decimal: str, converters: Optional[Dict], na_values: Optional[Any], keep_default_na: bool, displayed_only: bool) -> List[DataFrame]` |
| `pandas.namespace.read_json` | Function | `(path: str, lines: bool, index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.namespace.read_orc` | Function | `(path: str, columns: Optional[List[str]], index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.namespace.read_parquet` | Function | `(path: str, columns: Optional[List[str]], index_col: Optional[List[str]], pandas_metadata: bool, options: Any) -> DataFrame` |
| `pandas.namespace.read_spark_io` | Function | `(path: Optional[str], format: Optional[str], schema: Union[str, StructType], index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.namespace.read_sql` | Function | `(sql: str, con: str, index_col: Optional[Union[str, List[str]]], columns: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.namespace.read_sql_query` | Function | `(sql: str, con: str, index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.namespace.read_sql_table` | Function | `(table_name: str, con: str, schema: Optional[str], index_col: Optional[Union[str, List[str]]], columns: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.namespace.read_table` | Function | `(name: str, index_col: Optional[Union[str, List[str]]]) -> DataFrame` |
| `pandas.namespace.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.namespace.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.namespace.timedelta_range` | Function | `(start: Union[str, Any], end: Union[str, Any], periods: Optional[int], freq: Optional[Union[str, DateOffset]], name: Optional[str], closed: Optional[str]) -> TimedeltaIndex` |
| `pandas.namespace.to_datetime` | Function | `(arg, errors: str, format: Optional[str], unit: Optional[str], infer_datetime_format: bool, origin: str)` |
| `pandas.namespace.to_numeric` | Function | `(arg, errors)` |
| `pandas.namespace.to_timedelta` | Function | `(arg, unit: Optional[str], errors: str)` |
| `pandas.namespace.validate_axis` | Function | `(axis: Optional[Axis], none_axis: int) -> int` |
| `pandas.notna` | Function | `(obj)` |
| `pandas.notnull` | Object | `` |
| `pandas.numpy_compat.BooleanType` | Class | `(...)` |
| `pandas.numpy_compat.DoubleType` | Class | `(...)` |
| `pandas.numpy_compat.F` | Object | `` |
| `pandas.numpy_compat.IndexOpsMixin` | Class | `(...)` |
| `pandas.numpy_compat.LongType` | Class | `(...)` |
| `pandas.numpy_compat.binary_np_spark_mappings` | Object | `` |
| `pandas.numpy_compat.maybe_dispatch_ufunc_to_dunder_op` | Function | `(ser_or_index: IndexOpsMixin, ufunc: Callable, method: str, inputs: Any, kwargs: Any) -> IndexOpsMixin` |
| `pandas.numpy_compat.maybe_dispatch_ufunc_to_spark_func` | Function | `(ser_or_index: IndexOpsMixin, ufunc: Callable, method: str, inputs: Any, kwargs: Any) -> IndexOpsMixin` |
| `pandas.numpy_compat.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `pandas.numpy_compat.unary_np_spark_mappings` | Object | `` |
| `pandas.option_context` | Function | `(args: Any) -> Iterator[None]` |
| `pandas.options` | Object | `` |
| `pandas.plot.BoxPlotBase` | Class | `(...)` |
| `pandas.plot.Column` | Class | `(...)` |
| `pandas.plot.F` | Object | `` |
| `pandas.plot.HistogramPlotBase` | Class | `(...)` |
| `pandas.plot.KdePlotBase` | Class | `(...)` |
| `pandas.plot.NumericPlotBase` | Class | `(...)` |
| `pandas.plot.PandasObject` | Object | `` |
| `pandas.plot.PandasOnSparkPlotAccessor` | Class | `(data)` |
| `pandas.plot.SF` | Class | `(...)` |
| `pandas.plot.SampledPlotBase` | Class | `(...)` |
| `pandas.plot.TopNPlotBase` | Class | `(...)` |
| `pandas.plot.core.BoxPlotBase` | Class | `(...)` |
| `pandas.plot.core.Column` | Class | `(...)` |
| `pandas.plot.core.F` | Object | `` |
| `pandas.plot.core.HistogramPlotBase` | Class | `(...)` |
| `pandas.plot.core.KdePlotBase` | Class | `(...)` |
| `pandas.plot.core.NumericPlotBase` | Class | `(...)` |
| `pandas.plot.core.PandasOnSparkPlotAccessor` | Class | `(data)` |
| `pandas.plot.core.SF` | Class | `(...)` |
| `pandas.plot.core.SampledPlotBase` | Class | `(...)` |
| `pandas.plot.core.TopNPlotBase` | Class | `(...)` |
| `pandas.plot.core.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.plot.core.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.plot.core.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.plot.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.plot.importlib` | Object | `` |
| `pandas.plot.is_integer` | Object | `` |
| `pandas.plot.math` | Object | `` |
| `pandas.plot.matplotlib.BoxPlotBase` | Class | `(...)` |
| `pandas.plot.matplotlib.HistogramPlotBase` | Class | `(...)` |
| `pandas.plot.matplotlib.KdePlotBase` | Class | `(...)` |
| `pandas.plot.matplotlib.LooseVersion` | Class | `(vstring: Optional[str])` |
| `pandas.plot.matplotlib.PandasOnSparkAreaPlot` | Class | `(data, kwargs)` |
| `pandas.plot.matplotlib.PandasOnSparkBarPlot` | Class | `(data, kwargs)` |
| `pandas.plot.matplotlib.PandasOnSparkBarhPlot` | Class | `(data, kwargs)` |
| `pandas.plot.matplotlib.PandasOnSparkBoxPlot` | Class | `(...)` |
| `pandas.plot.matplotlib.PandasOnSparkHistPlot` | Class | `(...)` |
| `pandas.plot.matplotlib.PandasOnSparkKdePlot` | Class | `(...)` |
| `pandas.plot.matplotlib.PandasOnSparkLinePlot` | Class | `(data, kwargs)` |
| `pandas.plot.matplotlib.PandasOnSparkPiePlot` | Class | `(data, kwargs)` |
| `pandas.plot.matplotlib.PandasOnSparkScatterPlot` | Class | `(data, x, y, kwargs)` |
| `pandas.plot.matplotlib.SampledPlotBase` | Class | `(...)` |
| `pandas.plot.matplotlib.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.plot.matplotlib.TopNPlotBase` | Class | `(...)` |
| `pandas.plot.matplotlib.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.plot.matplotlib.plot_frame` | Function | `(data, x, y, kind, ax, subplots, sharex, sharey, layout, figsize, use_index, title, grid, legend, style, logx, logy, loglog, xticks, yticks, xlim, ylim, rot, fontsize, colormap, table, yerr, xerr, secondary_y, kwds)` |
| `pandas.plot.matplotlib.plot_pandas_on_spark` | Function | `(data, kind, kwargs)` |
| `pandas.plot.matplotlib.plot_series` | Function | `(data, kind, ax, figsize, use_index, title, grid, legend, style, logx, logy, loglog, xticks, yticks, xlim, ylim, rot, fontsize, colormap, table, yerr, xerr, label, secondary_y, kwds)` |
| `pandas.plot.matplotlib.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.plot.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.plot.np` | Object | `` |
| `pandas.plot.pd` | Object | `` |
| `pandas.plot.plotly.BoxPlotBase` | Class | `(...)` |
| `pandas.plot.plotly.HistogramPlotBase` | Class | `(...)` |
| `pandas.plot.plotly.KdePlotBase` | Class | `(...)` |
| `pandas.plot.plotly.PandasOnSparkPlotAccessor` | Class | `(data)` |
| `pandas.plot.plotly.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.plot.plotly.plot_box` | Function | `(data: Union[ps.DataFrame, ps.Series], kwargs)` |
| `pandas.plot.plotly.plot_histogram` | Function | `(data: Union[ps.DataFrame, ps.Series], kwargs)` |
| `pandas.plot.plotly.plot_kde` | Function | `(data: Union[ps.DataFrame, ps.Series], kwargs)` |
| `pandas.plot.plotly.plot_pandas_on_spark` | Function | `(data: Union[ps.DataFrame, ps.Series], kind: str, kwargs)` |
| `pandas.plot.plotly.plot_pie` | Function | `(data: Union[ps.DataFrame, ps.Series], kwargs)` |
| `pandas.plot.plotly.ps` | Object | `` |
| `pandas.plot.unsupported_function` | Function | `(class_name, method_name, deprecated, reason)` |
| `pandas.range` | Function | `(start: int, end: Optional[int], step: int, num_partitions: Optional[int]) -> DataFrame` |
| `pandas.read_clipboard` | Function | `(sep: str, kwargs: Any) -> DataFrame` |
| `pandas.read_csv` | Function | `(path: Union[str, List[str]], sep: str, header: Union[str, int, None], names: Optional[Union[str, List[str]]], index_col: Optional[Union[str, List[str]]], usecols: Optional[Union[List[int], List[str], Callable[[str], bool]]], dtype: Optional[Union[str, Dtype, Dict[str, Union[str, Dtype]]]], nrows: Optional[int], parse_dates: bool, quotechar: Optional[str], escapechar: Optional[str], comment: Optional[str], encoding: Optional[str], options: Any) -> Union[DataFrame, Series]` |
| `pandas.read_delta` | Function | `(path: str, version: Optional[str], timestamp: Optional[str], index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.read_excel` | Function | `(io: Union[str, Any], sheet_name: Union[str, int, List[Union[str, int]], None], header: Union[int, List[int]], names: Optional[List], index_col: Optional[List[int]], usecols: Optional[Union[int, str, List[Union[int, str]], Callable[[str], bool]]], dtype: Optional[Dict[str, Union[str, Dtype]]], engine: Optional[str], converters: Optional[Dict], true_values: Optional[Any], false_values: Optional[Any], skiprows: Optional[Union[int, List[int]]], nrows: Optional[int], na_values: Optional[Any], keep_default_na: bool, verbose: bool, parse_dates: Union[bool, List, Dict], date_parser: Optional[Callable], thousands: Optional[str], comment: Optional[str], skipfooter: int, kwds: Any) -> Union[DataFrame, Series, Dict[str, Union[DataFrame, Series]]]` |
| `pandas.read_html` | Function | `(io: Union[str, Any], match: str, flavor: Optional[str], header: Optional[Union[int, List[int]]], index_col: Optional[Union[int, List[int]]], skiprows: Optional[Union[int, List[int], slice]], attrs: Optional[Dict[str, str]], parse_dates: bool, thousands: str, encoding: Optional[str], decimal: str, converters: Optional[Dict], na_values: Optional[Any], keep_default_na: bool, displayed_only: bool) -> List[DataFrame]` |
| `pandas.read_json` | Function | `(path: str, lines: bool, index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.read_orc` | Function | `(path: str, columns: Optional[List[str]], index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.read_parquet` | Function | `(path: str, columns: Optional[List[str]], index_col: Optional[List[str]], pandas_metadata: bool, options: Any) -> DataFrame` |
| `pandas.read_spark_io` | Function | `(path: Optional[str], format: Optional[str], schema: Union[str, StructType], index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.read_sql` | Function | `(sql: str, con: str, index_col: Optional[Union[str, List[str]]], columns: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.read_sql_query` | Function | `(sql: str, con: str, index_col: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.read_sql_table` | Function | `(table_name: str, con: str, schema: Optional[str], index_col: Optional[Union[str, List[str]]], columns: Optional[Union[str, List[str]]], options: Any) -> DataFrame` |
| `pandas.read_table` | Function | `(name: str, index_col: Optional[Union[str, List[str]]]) -> DataFrame` |
| `pandas.require_minimum_pandas_version` | Function | `() -> None` |
| `pandas.require_minimum_pyarrow_version` | Function | `() -> None` |
| `pandas.resample.Column` | Class | `(...)` |
| `pandas.resample.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.resample.DataFrameResampler` | Class | `(psdf: DataFrame, resamplekey: Optional[Series], rule: str, closed: Optional[str], label: Optional[str], agg_columns: List[Series])` |
| `pandas.resample.DataType` | Class | `(...)` |
| `pandas.resample.F` | Object | `` |
| `pandas.resample.FrameLike` | Object | `` |
| `pandas.resample.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.resample.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.resample.MissingPandasLikeDataFrameResampler` | Class | `(...)` |
| `pandas.resample.MissingPandasLikeSeriesResampler` | Class | `(...)` |
| `pandas.resample.NumericType` | Class | `(...)` |
| `pandas.resample.Resampler` | Class | `(psdf: DataFrame, resamplekey: Optional[Series], rule: str, closed: Optional[str], label: Optional[str], agg_columns: List[Series])` |
| `pandas.resample.SF` | Class | `(...)` |
| `pandas.resample.SPARK_DEFAULT_INDEX_NAME` | Object | `` |
| `pandas.resample.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.resample.SeriesResampler` | Class | `(psser: Series, resamplekey: Optional[Series], rule: str, closed: Optional[str], label: Optional[str], agg_columns: List[Series])` |
| `pandas.resample.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.resample.TimestampNTZType` | Class | `(...)` |
| `pandas.resample.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.resample.ps` | Object | `` |
| `pandas.resample.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.resample.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.reset_option` | Function | `(key: str, spark_session: Optional[SparkSession]) -> None` |
| `pandas.series.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `pandas.series.Axis` | Object | `` |
| `pandas.series.BooleanType` | Class | `(...)` |
| `pandas.series.CORRELATION_CORR_OUTPUT_COLUMN` | Object | `` |
| `pandas.series.CORRELATION_COUNT_OUTPUT_COLUMN` | Object | `` |
| `pandas.series.CORRELATION_VALUE_1_COLUMN` | Object | `` |
| `pandas.series.CORRELATION_VALUE_2_COLUMN` | Object | `` |
| `pandas.series.CategoricalAccessor` | Class | `(series: ps.Series)` |
| `pandas.series.ColumnOrName` | Object | `` |
| `pandas.series.DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.series.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.series.DatetimeMethods` | Class | `(series: ps.Series)` |
| `pandas.series.DecimalType` | Class | `(precision: int, scale: int)` |
| `pandas.series.DoubleType` | Class | `(...)` |
| `pandas.series.Dtype` | Object | `` |
| `pandas.series.F` | Object | `` |
| `pandas.series.FloatType` | Class | `(...)` |
| `pandas.series.Frame` | Class | `(...)` |
| `pandas.series.FuncT` | Object | `` |
| `pandas.series.Index` | Class | `(...)` |
| `pandas.series.IndexOpsMixin` | Class | `(...)` |
| `pandas.series.IntegerType` | Class | `(...)` |
| `pandas.series.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.series.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.series.Label` | Object | `` |
| `pandas.series.LongType` | Class | `(...)` |
| `pandas.series.MissingPandasLikeSeries` | Class | `(...)` |
| `pandas.series.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.series.Name` | Object | `` |
| `pandas.series.NullType` | Class | `(...)` |
| `pandas.series.NumericType` | Class | `(...)` |
| `pandas.series.PandasOnSparkPlotAccessor` | Class | `(data)` |
| `pandas.series.PandasOnSparkSeriesMethods` | Class | `(series: Series)` |
| `pandas.series.PySparkColumn` | Class | `(...)` |
| `pandas.series.PySparkWindow` | Class | `(...)` |
| `pandas.series.REPR_PATTERN` | Object | `` |
| `pandas.series.Row` | Class | `(...)` |
| `pandas.series.SF` | Class | `(...)` |
| `pandas.series.SPARK_CONF_ARROW_ENABLED` | Object | `` |
| `pandas.series.SPARK_DEFAULT_INDEX_NAME` | Object | `` |
| `pandas.series.SPARK_DEFAULT_SERIES_NAME` | Object | `` |
| `pandas.series.Scalar` | Object | `` |
| `pandas.series.ScalarType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.series.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.series.SeriesGroupBy` | Class | `(psser: Series, by: List[Series], as_index: bool, dropna: bool)` |
| `pandas.series.SeriesResampler` | Class | `(psser: Series, resamplekey: Optional[Series], rule: str, closed: Optional[str], label: Optional[str], agg_columns: List[Series])` |
| `pandas.series.SeriesType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.series.SparkDataFrame` | Class | `(...)` |
| `pandas.series.SparkIndexOpsMethods` | Class | `(data: IndexOpsLike)` |
| `pandas.series.SparkPandasIndexingError` | Class | `(...)` |
| `pandas.series.SparkSeriesMethods` | Class | `(...)` |
| `pandas.series.StringMethods` | Class | `(series: ps.Series)` |
| `pandas.series.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.series.T` | Object | `` |
| `pandas.series.TimestampType` | Class | `(...)` |
| `pandas.series.Window` | Class | `(...)` |
| `pandas.series.ansi_mode_context` | Function | `(spark: SparkSession) -> Iterator[None]` |
| `pandas.series.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.series.combine_frames` | Function | `(this: DataFrame, args: DataFrameOrSeries, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.series.compute` | Function | `(sdf: SparkDataFrame, groupKeys: List[str], method: str) -> SparkDataFrame` |
| `pandas.series.create_type_for_series_type` | Function | `(param: Any) -> Type[SeriesType]` |
| `pandas.series.first_series` | Function | `(df: Union[DataFrame, pd.DataFrame]) -> Union[Series, pd.Series]` |
| `pandas.series.get_option` | Function | `(key: str, default: Union[Any, _NoValueType], spark_session: Optional[SparkSession]) -> Any` |
| `pandas.series.infer_return_type` | Function | `(f: Callable) -> Union[SeriesType, DataFrameType, ScalarType, UnknownType]` |
| `pandas.series.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.series.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.series.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.series.log_advice` | Function | `(message: str) -> None` |
| `pandas.series.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.series.ps` | Object | `` |
| `pandas.series.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.series.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.series.spark_type_to_pandas_dtype` | Function | `(spark_type: types.DataType, use_extension_dtypes: bool) -> Dtype` |
| `pandas.series.sql_conf` | Function | `(pairs: Dict[str, Any], spark: Optional[SparkSession]) -> Iterator[None]` |
| `pandas.series.str_type` | Object | `` |
| `pandas.series.unpack_scalar` | Function | `(sdf: SparkDataFrame) -> Any` |
| `pandas.series.validate_arguments_and_invoke_function` | Function | `(pobj: Union[pd.DataFrame, pd.Series], pandas_on_spark_func: Callable, pandas_func: Callable, input_args: Dict) -> Any` |
| `pandas.series.validate_axis` | Function | `(axis: Optional[Axis], none_axis: int) -> int` |
| `pandas.series.validate_bool_kwarg` | Function | `(value: Any, arg_name: str) -> Optional[bool]` |
| `pandas.series.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.series.with_ansi_mode_context` | Function | `(f: FuncT) -> FuncT` |
| `pandas.set_option` | Function | `(key: str, value: Any, spark_session: Optional[SparkSession]) -> None` |
| `pandas.spark.accessors.CachedDataFrame` | Class | `(internal: InternalFrame, storage_level: Optional[StorageLevel])` |
| `pandas.spark.accessors.CachedSparkFrameMethods` | Class | `(frame: CachedDataFrame)` |
| `pandas.spark.accessors.DataType` | Class | `(...)` |
| `pandas.spark.accessors.IndexOpsLike` | Object | `` |
| `pandas.spark.accessors.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.spark.accessors.OptionalPrimitiveType` | Object | `` |
| `pandas.spark.accessors.PrimitiveType` | Object | `` |
| `pandas.spark.accessors.PySparkColumn` | Class | `(...)` |
| `pandas.spark.accessors.PySparkDataFrame` | Class | `(...)` |
| `pandas.spark.accessors.SparkFrameMethods` | Class | `(frame: ps.DataFrame)` |
| `pandas.spark.accessors.SparkIndexMethods` | Class | `(...)` |
| `pandas.spark.accessors.SparkIndexOpsMethods` | Class | `(data: IndexOpsLike)` |
| `pandas.spark.accessors.SparkSeriesMethods` | Class | `(...)` |
| `pandas.spark.accessors.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `pandas.spark.accessors.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.spark.accessors.ps` | Object | `` |
| `pandas.spark.utils.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `pandas.spark.utils.DataType` | Class | `(...)` |
| `pandas.spark.utils.DecimalType` | Class | `(precision: int, scale: int)` |
| `pandas.spark.utils.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `pandas.spark.utils.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `pandas.spark.utils.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pandas.spark.utils.as_nullable_spark_type` | Function | `(dt: DataType) -> DataType` |
| `pandas.spark.utils.force_decimal_precision_scale` | Function | `(dt: DataType, precision: int, scale: int) -> DataType` |
| `pandas.sql` | Function | `(query: str, index_col: Optional[Union[str, List[str]]], args: Optional[Union[Dict[str, Any], List]], kwargs: Any) -> DataFrame` |
| `pandas.sql_formatter.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.sql_formatter.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.sql_formatter.PandasSQLStringFormatter` | Class | `(session: SparkSession)` |
| `pandas.sql_formatter.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.sql_formatter.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pandas.sql_formatter.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.sql_formatter.get_lit_sql_str` | Function | `(val: str) -> str` |
| `pandas.sql_formatter.is_remote` | Function | `() -> bool` |
| `pandas.sql_formatter.ps` | Object | `` |
| `pandas.sql_formatter.sql` | Function | `(query: str, index_col: Optional[Union[str, List[str]]], args: Optional[Union[Dict[str, Any], List]], kwargs: Any) -> DataFrame` |
| `pandas.sql_processor.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.sql_processor.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.sql_processor.SDataFrame` | Class | `(...)` |
| `pandas.sql_processor.SQLProcessor` | Class | `(scope: Dict[str, Any], statement: str, session: SparkSession)` |
| `pandas.sql_processor.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.sql_processor.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pandas.sql_processor.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.sql_processor.escape_sql_string` | Function | `(value: str) -> str` |
| `pandas.sql_processor.ps` | Object | `` |
| `pandas.sql_processor.sql` | Function | `(query: str, index_col: Optional[Union[str, List[str]]], globals: Optional[Dict[str, Any]], locals: Optional[Dict[str, Any]], kwargs: Any) -> DataFrame` |
| `pandas.strings.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `pandas.strings.BinaryType` | Class | `(...)` |
| `pandas.strings.F` | Object | `` |
| `pandas.strings.FuncT` | Object | `` |
| `pandas.strings.LongType` | Class | `(...)` |
| `pandas.strings.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `pandas.strings.StringMethods` | Class | `(series: ps.Series)` |
| `pandas.strings.StringType` | Class | `(collation: str)` |
| `pandas.strings.ansi_mode_context` | Function | `(spark: SparkSession) -> Iterator[None]` |
| `pandas.strings.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.strings.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `pandas.strings.ps` | Object | `` |
| `pandas.strings.with_ansi_mode_context` | Function | `(f: FuncT) -> FuncT` |
| `pandas.supported_api_gen.COMMON_PARAMETER_SET` | Object | `` |
| `pandas.supported_api_gen.Implemented` | Class | `(...)` |
| `pandas.supported_api_gen.LooseVersion` | Class | `(vstring: Optional[str])` |
| `pandas.supported_api_gen.MAX_MISSING_PARAMS_SIZE` | Object | `` |
| `pandas.supported_api_gen.MODULE_GROUP_MATCH` | Object | `` |
| `pandas.supported_api_gen.PANDAS_LATEST_VERSION` | Object | `` |
| `pandas.supported_api_gen.PandasNotImplementedError` | Class | `(class_name: str, method_name: Optional[str], arg_name: Optional[str], property_name: Optional[str], scalar_name: Optional[str], deprecated: bool, reason: str)` |
| `pandas.supported_api_gen.RST_HEADER` | Object | `` |
| `pandas.supported_api_gen.SupportedStatus` | Class | `(...)` |
| `pandas.supported_api_gen.generate_supported_api` | Function | `(output_rst_file_path: str) -> None` |
| `pandas.supported_api_gen.ps` | Object | `` |
| `pandas.supported_api_gen.psg` | Object | `` |
| `pandas.supported_api_gen.psw` | Object | `` |
| `pandas.testing.assert_frame_equal` | Function | `(left: Union[ps.DataFrame, pd.DataFrame], right: Union[ps.DataFrame, pd.DataFrame], check_dtype: bool, check_index_type: Union[bool, Literal['equiv']], check_column_type: Union[bool, Literal['equiv']], check_frame_type: bool, check_names: bool, by_blocks: bool, check_exact: bool, check_datetimelike_compat: bool, check_categorical: bool, check_like: bool, check_freq: bool, check_flags: bool, rtol: float, atol: float, obj: str) -> None` |
| `pandas.testing.assert_index_equal` | Function | `(left: Union[ps.Index, pd.Index], right: Union[ps.Index, pd.Index], exact: Union[bool, Literal['equiv']], check_names: bool, check_exact: bool, check_categorical: bool, check_order: bool, rtol: float, atol: float, obj: str) -> None` |
| `pandas.testing.assert_series_equal` | Function | `(left: Union[ps.Series, pd.Series], right: Union[ps.Series, pd.Series], check_dtype: bool, check_index_type: Union[bool, Literal['equiv']], check_series_type: bool, check_names: bool, check_exact: bool, check_datetimelike_compat: bool, check_categorical: bool, check_category_order: bool, check_freq: bool, check_flags: bool, rtol: float, atol: float, obj: str, check_index: bool, check_like: bool) -> None` |
| `pandas.testing.ps` | Object | `` |
| `pandas.testing.require_minimum_pandas_version` | Function | `() -> None` |
| `pandas.timedelta_range` | Function | `(start: Union[str, Any], end: Union[str, Any], periods: Optional[int], freq: Optional[Union[str, DateOffset]], name: Optional[str], closed: Optional[str]) -> TimedeltaIndex` |
| `pandas.to_datetime` | Function | `(arg, errors: str, format: Optional[str], unit: Optional[str], infer_datetime_format: bool, origin: str)` |
| `pandas.to_numeric` | Function | `(arg, errors)` |
| `pandas.to_timedelta` | Function | `(arg, unit: Optional[str], errors: str)` |
| `pandas.typedef.Any` | Object | `` |
| `pandas.typedef.BooleanDtype` | Object | `` |
| `pandas.typedef.Callable` | Object | `` |
| `pandas.typedef.CategoricalDtype` | Object | `` |
| `pandas.typedef.DataFrameType` | Class | `(index_fields: List[InternalField], data_fields: List[InternalField])` |
| `pandas.typedef.Dtype` | Object | `` |
| `pandas.typedef.ExtensionDtype` | Object | `` |
| `pandas.typedef.Float32Dtype` | Object | `` |
| `pandas.typedef.Float64Dtype` | Object | `` |
| `pandas.typedef.Generic` | Object | `` |
| `pandas.typedef.IndexNameTypeHolder` | Class | `(...)` |
| `pandas.typedef.Int16Dtype` | Object | `` |
| `pandas.typedef.Int32Dtype` | Object | `` |
| `pandas.typedef.Int64Dtype` | Object | `` |
| `pandas.typedef.Int8Dtype` | Object | `` |
| `pandas.typedef.Iterable` | Object | `` |
| `pandas.typedef.List` | Object | `` |
| `pandas.typedef.NameTypeHolder` | Class | `(...)` |
| `pandas.typedef.ScalarType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.typedef.SeriesType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.typedef.StringDtype` | Object | `` |
| `pandas.typedef.T` | Object | `` |
| `pandas.typedef.Tuple` | Object | `` |
| `pandas.typedef.Type` | Object | `` |
| `pandas.typedef.Union` | Object | `` |
| `pandas.typedef.UnknownType` | Class | `(tpe: Any)` |
| `pandas.typedef.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.typedef.create_tuple_for_frame_type` | Function | `(params: Any) -> object` |
| `pandas.typedef.create_type_for_series_type` | Function | `(param: Any) -> Type[SeriesType]` |
| `pandas.typedef.datetime` | Object | `` |
| `pandas.typedef.decimal` | Object | `` |
| `pandas.typedef.extension_dtypes` | Object | `` |
| `pandas.typedef.extension_dtypes_available` | Object | `` |
| `pandas.typedef.extension_float_dtypes_available` | Object | `` |
| `pandas.typedef.extension_object_dtypes_available` | Object | `` |
| `pandas.typedef.from_arrow_type` | Function | `(at: pa.DataType, prefer_timestamp_ntz: bool) -> DataType` |
| `pandas.typedef.get_type_hints` | Object | `` |
| `pandas.typedef.infer_pd_series_spark_type` | Function | `(pser: pd.Series, dtype: Dtype, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.typedef.infer_return_type` | Function | `(f: Callable) -> Union[SeriesType, DataFrameType, ScalarType, UnknownType]` |
| `pandas.typedef.isclass` | Object | `` |
| `pandas.typedef.np` | Object | `` |
| `pandas.typedef.pa` | Object | `` |
| `pandas.typedef.pandas_dtype` | Object | `` |
| `pandas.typedef.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.typedef.pd` | Object | `` |
| `pandas.typedef.ps` | Object | `` |
| `pandas.typedef.spark_type_to_pandas_dtype` | Function | `(spark_type: types.DataType, use_extension_dtypes: bool) -> Dtype` |
| `pandas.typedef.sys` | Object | `` |
| `pandas.typedef.to_arrow_type` | Function | `(dt: DataType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.DataType` |
| `pandas.typedef.typehints.DataFrameType` | Class | `(index_fields: List[InternalField], data_fields: List[InternalField])` |
| `pandas.typedef.typehints.Dtype` | Object | `` |
| `pandas.typedef.typehints.IndexNameTypeHolder` | Class | `(...)` |
| `pandas.typedef.typehints.InternalField` | Class | `(dtype: Dtype, struct_field: Optional[StructField])` |
| `pandas.typedef.typehints.NameTypeHolder` | Class | `(...)` |
| `pandas.typedef.typehints.ScalarType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.typedef.typehints.SeriesType` | Class | `(dtype: Dtype, spark_type: types.DataType)` |
| `pandas.typedef.typehints.T` | Object | `` |
| `pandas.typedef.typehints.UnknownType` | Class | `(tpe: Any)` |
| `pandas.typedef.typehints.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.typedef.typehints.create_tuple_for_frame_type` | Function | `(params: Any) -> object` |
| `pandas.typedef.typehints.create_type_for_series_type` | Function | `(param: Any) -> Type[SeriesType]` |
| `pandas.typedef.typehints.extension_dtypes` | Object | `` |
| `pandas.typedef.typehints.extension_dtypes_available` | Object | `` |
| `pandas.typedef.typehints.extension_float_dtypes_available` | Object | `` |
| `pandas.typedef.typehints.extension_object_dtypes_available` | Object | `` |
| `pandas.typedef.typehints.from_arrow_type` | Function | `(at: pa.DataType, prefer_timestamp_ntz: bool) -> DataType` |
| `pandas.typedef.typehints.infer_pd_series_spark_type` | Function | `(pser: pd.Series, dtype: Dtype, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.typedef.typehints.infer_return_type` | Function | `(f: Callable) -> Union[SeriesType, DataFrameType, ScalarType, UnknownType]` |
| `pandas.typedef.typehints.pandas_on_spark_type` | Function | `(tpe: Union[str, type, Dtype]) -> Tuple[Dtype, types.DataType]` |
| `pandas.typedef.typehints.ps` | Object | `` |
| `pandas.typedef.typehints.spark_type_to_pandas_dtype` | Function | `(spark_type: types.DataType, use_extension_dtypes: bool) -> Dtype` |
| `pandas.typedef.typehints.to_arrow_type` | Function | `(dt: DataType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.DataType` |
| `pandas.typedef.typehints.types` | Object | `` |
| `pandas.typedef.types` | Object | `` |
| `pandas.typedef.typing` | Object | `` |
| `pandas.usage_logging.CachedSparkFrameMethods` | Class | `(frame: CachedDataFrame)` |
| `pandas.usage_logging.CategoricalIndex` | Class | `(...)` |
| `pandas.usage_logging.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.usage_logging.DataFrameGroupBy` | Class | `(psdf: DataFrame, by: List[Series], as_index: bool, dropna: bool, column_labels_to_exclude: Set[Label], agg_columns: List[Label])` |
| `pandas.usage_logging.DatetimeIndex` | Class | `(...)` |
| `pandas.usage_logging.DatetimeMethods` | Class | `(series: ps.Series)` |
| `pandas.usage_logging.Expanding` | Class | `(psdf_or_psser: FrameLike, min_periods: int)` |
| `pandas.usage_logging.ExpandingGroupby` | Class | `(groupby: GroupBy[FrameLike], min_periods: int)` |
| `pandas.usage_logging.ExponentialMoving` | Class | `(psdf_or_psser: FrameLike, com: Optional[float], span: Optional[float], halflife: Optional[float], alpha: Optional[float], min_periods: Optional[int], ignore_na: bool)` |
| `pandas.usage_logging.ExponentialMovingGroupby` | Class | `(groupby: GroupBy[FrameLike], com: Optional[float], span: Optional[float], halflife: Optional[float], alpha: Optional[float], min_periods: Optional[int], ignore_na: bool)` |
| `pandas.usage_logging.Index` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeDataFrame` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeDataFrameGroupBy` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeDatetimeIndex` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeExpanding` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeExpandingGroupby` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeExponentialMoving` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeExponentialMovingGroupby` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeGeneralFunctions` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeIndex` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeMultiIndex` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeRolling` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeRollingGroupby` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeSeries` | Class | `(...)` |
| `pandas.usage_logging.MissingPandasLikeSeriesGroupBy` | Class | `(...)` |
| `pandas.usage_logging.MultiIndex` | Class | `(...)` |
| `pandas.usage_logging.PandasOnSparkFrameMethods` | Class | `(frame: DataFrame)` |
| `pandas.usage_logging.Rolling` | Class | `(psdf_or_psser: FrameLike, window: int, min_periods: Optional[int])` |
| `pandas.usage_logging.RollingGroupby` | Class | `(groupby: GroupBy[FrameLike], window: int, min_periods: Optional[int])` |
| `pandas.usage_logging.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.usage_logging.SeriesGroupBy` | Class | `(psser: Series, by: List[Series], as_index: bool, dropna: bool)` |
| `pandas.usage_logging.SparkFrameMethods` | Class | `(frame: ps.DataFrame)` |
| `pandas.usage_logging.SparkIndexOpsMethods` | Class | `(data: IndexOpsLike)` |
| `pandas.usage_logging.StringMethods` | Class | `(series: ps.Series)` |
| `pandas.usage_logging.attach` | Function | `(logger_module: Union[str, ModuleType]) -> None` |
| `pandas.usage_logging.config` | Object | `` |
| `pandas.usage_logging.namespace` | Object | `` |
| `pandas.usage_logging.sql_formatter` | Object | `` |
| `pandas.usage_logging.usage_logger.PandasOnSparkUsageLogger` | Class | `()` |
| `pandas.usage_logging.usage_logger.get_logger` | Function | `() -> Any` |
| `pandas.utils.Axis` | Object | `` |
| `pandas.utils.Column` | Class | `(...)` |
| `pandas.utils.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `pandas.utils.DataFrameOrSeries` | Object | `` |
| `pandas.utils.DoubleType` | Class | `(...)` |
| `pandas.utils.ERROR_MESSAGE_CANNOT_COMBINE` | Object | `` |
| `pandas.utils.F` | Object | `` |
| `pandas.utils.Index` | Class | `(...)` |
| `pandas.utils.IndexOpsMixin` | Class | `(...)` |
| `pandas.utils.InternalFrame` | Class | `(spark_frame: PySparkDataFrame, index_spark_columns: Optional[List[PySparkColumn]], index_names: Optional[List[Optional[Label]]], index_fields: Optional[List[InternalField]], column_labels: Optional[List[Label]], data_spark_columns: Optional[List[PySparkColumn]], data_fields: Optional[List[InternalField]], column_label_names: Optional[List[Optional[Label]]])` |
| `pandas.utils.Label` | Object | `` |
| `pandas.utils.Name` | Object | `` |
| `pandas.utils.PandasAPIOnSparkAdviceWarning` | Class | `(...)` |
| `pandas.utils.PySparkDataFrame` | Class | `(...)` |
| `pandas.utils.PySparkTypeError` | Class | `(...)` |
| `pandas.utils.SPARK_CONF_ARROW_ENABLED` | Object | `` |
| `pandas.utils.SPARK_CONF_PANDAS_STRUCT_MODE` | Object | `` |
| `pandas.utils.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `pandas.utils.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pandas.utils.UnsupportedOperationException` | Class | `(...)` |
| `pandas.utils.align_diff_frames` | Function | `(resolve_func: Callable[[DataFrame, List[Label], List[Label]], Iterator[Tuple[Series, Label]]], this: DataFrame, that: DataFrame, fillna: bool, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.utils.ansi_mode_context` | Function | `(spark: SparkSession) -> Iterator[None]` |
| `pandas.utils.as_spark_type` | Function | `(tpe: Union[str, type, Dtype], raise_error: bool, prefer_timestamp_ntz: bool) -> types.DataType` |
| `pandas.utils.column_labels_level` | Function | `(column_labels: List[Label]) -> int` |
| `pandas.utils.combine_frames` | Function | `(this: DataFrame, args: DataFrameOrSeries, how: str, preserve_order_column: bool) -> DataFrame` |
| `pandas.utils.compare_allow_null` | Function | `(left: Column, right: Column, comp: Callable[[Column, Column], Column]) -> Column` |
| `pandas.utils.compare_disallow_null` | Function | `(left: Column, right: Column, comp: Callable[[Column, Column], Column]) -> Column` |
| `pandas.utils.compare_null_first` | Function | `(left: Column, right: Column, comp: Callable[[Column, Column], Column]) -> Column` |
| `pandas.utils.compare_null_last` | Function | `(left: Column, right: Column, comp: Callable[[Column, Column], Column]) -> Column` |
| `pandas.utils.default_session` | Function | `(check_ansi_mode: bool) -> SparkSession` |
| `pandas.utils.is_ansi_mode_enabled` | Function | `(spark: SparkSession) -> bool` |
| `pandas.utils.is_name_like_tuple` | Function | `(value: Any, allow_none: bool, check_type: bool) -> bool` |
| `pandas.utils.is_name_like_value` | Function | `(value: Any, allow_none: bool, allow_tuple: bool, check_type: bool) -> bool` |
| `pandas.utils.is_remote` | Function | `() -> bool` |
| `pandas.utils.is_testing` | Function | `() -> bool` |
| `pandas.utils.lazy_property` | Function | `(fn: Callable[[Any], Any]) -> property` |
| `pandas.utils.log_advice` | Function | `(message: str) -> None` |
| `pandas.utils.name_like_string` | Function | `(name: Optional[Name]) -> str` |
| `pandas.utils.ps` | Object | `` |
| `pandas.utils.same_anchor` | Function | `(this: Union[DataFrame, IndexOpsMixin, InternalFrame], that: Union[DataFrame, IndexOpsMixin, InternalFrame]) -> bool` |
| `pandas.utils.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pandas.utils.spark_column_equals` | Function | `(left: Column, right: Column) -> bool` |
| `pandas.utils.sql_conf` | Function | `(pairs: Dict[str, Any], spark: Optional[SparkSession]) -> Iterator[None]` |
| `pandas.utils.validate_arguments_and_invoke_function` | Function | `(pobj: Union[pd.DataFrame, pd.Series], pandas_on_spark_func: Callable, pandas_func: Callable, input_args: Dict) -> Any` |
| `pandas.utils.validate_axis` | Function | `(axis: Optional[Axis], none_axis: int) -> int` |
| `pandas.utils.validate_bool_kwarg` | Function | `(value: Any, arg_name: str) -> Optional[bool]` |
| `pandas.utils.validate_how` | Function | `(how: str) -> str` |
| `pandas.utils.validate_index_loc` | Function | `(index: Index, loc: int) -> None` |
| `pandas.utils.validate_mode` | Function | `(mode: str) -> str` |
| `pandas.utils.verify_temp_column_name` | Function | `(df: Union[DataFrame, PySparkDataFrame], column_name_or_label: Union[str, Name]) -> Union[str, Label]` |
| `pandas.utils.xor` | Function | `(df1: PySparkDataFrame, df2: PySparkDataFrame) -> PySparkDataFrame` |
| `pandas.window.Column` | Class | `(...)` |
| `pandas.window.DataFrameGroupBy` | Class | `(psdf: DataFrame, by: List[Series], as_index: bool, dropna: bool, column_labels_to_exclude: Set[Label], agg_columns: List[Label])` |
| `pandas.window.DoubleType` | Class | `(...)` |
| `pandas.window.Expanding` | Class | `(psdf_or_psser: FrameLike, min_periods: int)` |
| `pandas.window.ExpandingGroupby` | Class | `(groupby: GroupBy[FrameLike], min_periods: int)` |
| `pandas.window.ExpandingLike` | Class | `(min_periods: int)` |
| `pandas.window.ExponentialMoving` | Class | `(psdf_or_psser: FrameLike, com: Optional[float], span: Optional[float], halflife: Optional[float], alpha: Optional[float], min_periods: Optional[int], ignore_na: bool)` |
| `pandas.window.ExponentialMovingGroupby` | Class | `(groupby: GroupBy[FrameLike], com: Optional[float], span: Optional[float], halflife: Optional[float], alpha: Optional[float], min_periods: Optional[int], ignore_na: bool)` |
| `pandas.window.ExponentialMovingLike` | Class | `(window: WindowSpec, com: Optional[float], span: Optional[float], halflife: Optional[float], alpha: Optional[float], min_periods: Optional[int], ignore_na: bool)` |
| `pandas.window.F` | Object | `` |
| `pandas.window.FrameLike` | Object | `` |
| `pandas.window.GroupBy` | Class | `(psdf: DataFrame, groupkeys: List[Series], as_index: bool, dropna: bool, column_labels_to_exclude: Set[Label], agg_columns_selected: bool, agg_columns: List[Series])` |
| `pandas.window.MissingPandasLikeExpanding` | Class | `(...)` |
| `pandas.window.MissingPandasLikeExpandingGroupby` | Class | `(...)` |
| `pandas.window.MissingPandasLikeExponentialMoving` | Class | `(...)` |
| `pandas.window.MissingPandasLikeExponentialMovingGroupby` | Class | `(...)` |
| `pandas.window.MissingPandasLikeRolling` | Class | `(...)` |
| `pandas.window.MissingPandasLikeRollingGroupby` | Class | `(...)` |
| `pandas.window.NATURAL_ORDER_COLUMN_NAME` | Object | `` |
| `pandas.window.Rolling` | Class | `(psdf_or_psser: FrameLike, window: int, min_periods: Optional[int])` |
| `pandas.window.RollingAndExpanding` | Class | `(window: WindowSpec, min_periods: int)` |
| `pandas.window.RollingGroupby` | Class | `(groupby: GroupBy[FrameLike], window: int, min_periods: Optional[int])` |
| `pandas.window.RollingLike` | Class | `(window: int, min_periods: Optional[int])` |
| `pandas.window.SF` | Class | `(...)` |
| `pandas.window.SPARK_INDEX_NAME_FORMAT` | Object | `` |
| `pandas.window.Window` | Class | `(...)` |
| `pandas.window.WindowSpec` | Class | `(...)` |
| `pandas.window.ps` | Object | `` |
| `pandas.window.scol_for` | Function | `(sdf: PySparkDataFrame, column_name: str) -> Column` |
| `pipelines.add_pipeline_analysis_context.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pipelines.add_pipeline_analysis_context.add_pipeline_analysis_context` | Function | `(spark: SparkSession, dataflow_graph_id: str, flow_name: Optional[str]) -> Generator[None, None, None]` |
| `pipelines.api.Flow` | Class | `(name: str, target: str, spark_conf: Dict[str, str], source_code_location: SourceCodeLocation, func: QueryFunction)` |
| `pipelines.api.MaterializedView` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.api.PySparkTypeError` | Class | `(...)` |
| `pipelines.api.QueryFunction` | Object | `` |
| `pipelines.api.Sink` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, format: str, options: Mapping[str, str])` |
| `pipelines.api.StreamingTable` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.api.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pipelines.api.TemporaryView` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation)` |
| `pipelines.api.append_flow` | Function | `(target: str, name: Optional[str], spark_conf: Optional[Dict[str, str]]) -> Callable[[QueryFunction], None]` |
| `pipelines.api.create_sink` | Function | `(name: str, format: str, options: Optional[Dict[str, str]]) -> None` |
| `pipelines.api.create_streaming_table` | Function | `(name: str, comment: Optional[str], table_properties: Optional[Dict[str, str]], partition_cols: Optional[List[str]], cluster_by: Optional[List[str]], schema: Optional[Union[StructType, str]], format: Optional[str]) -> None` |
| `pipelines.api.get_active_graph_element_registry` | Function | `() -> GraphElementRegistry` |
| `pipelines.api.get_caller_source_code_location` | Function | `(stacklevel: int) -> SourceCodeLocation` |
| `pipelines.api.materialized_view` | Function | `(query_function: Optional[QueryFunction], name: Optional[str], comment: Optional[str], spark_conf: Optional[Dict[str, str]], table_properties: Optional[Dict[str, str]], partition_cols: Optional[List[str]], cluster_by: Optional[List[str]], schema: Optional[Union[StructType, str]], format: Optional[str]) -> Union[Callable[[QueryFunction], None], None]` |
| `pipelines.api.table` | Function | `(query_function: Optional[QueryFunction], name: Optional[str], comment: Optional[str], spark_conf: Optional[Dict[str, str]], table_properties: Optional[Dict[str, str]], partition_cols: Optional[List[str]], cluster_by: Optional[List[str]], schema: Optional[Union[StructType, str]], format: Optional[str]) -> Union[Callable[[QueryFunction], None], None]` |
| `pipelines.api.temporary_view` | Function | `(query_function: Optional[QueryFunction], name: Optional[str], comment: Optional[str], spark_conf: Optional[Dict[str, str]]) -> Union[Callable[[QueryFunction], None], None]` |
| `pipelines.api.validate_optional_list_of_str_arg` | Function | `(arg_name: str, arg_value: Optional[List[str]]) -> None` |
| `pipelines.append_flow` | Function | `(target: str, name: Optional[str], spark_conf: Optional[Dict[str, str]]) -> Callable[[QueryFunction], None]` |
| `pipelines.block_connect_access.BLOCKED_RPC_NAMES` | Object | `` |
| `pipelines.block_connect_access.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `pipelines.block_connect_access.SparkConnectServiceStub` | Class | `(channel)` |
| `pipelines.block_connect_access.block_spark_connect_execution_and_analysis` | Function | `() -> Generator[None, None, None]` |
| `pipelines.block_session_mutations.BLOCKED_METHODS` | Object | `` |
| `pipelines.block_session_mutations.Catalog` | Class | `(sparkSession: SparkSession)` |
| `pipelines.block_session_mutations.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `pipelines.block_session_mutations.ERROR_CLASS` | Object | `` |
| `pipelines.block_session_mutations.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `pipelines.block_session_mutations.RuntimeConf` | Class | `(client: SparkConnectClient)` |
| `pipelines.block_session_mutations.UDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `pipelines.block_session_mutations.block_session_mutations` | Function | `() -> Generator[None, None, None]` |
| `pipelines.cli.GraphElementRegistry` | Class | `(...)` |
| `pipelines.cli.LibrariesGlob` | Class | `(include: str)` |
| `pipelines.cli.PIPELINE_SPEC_FILE_NAMES` | Object | `` |
| `pipelines.cli.PipelineSpec` | Class | `(name: str, storage: str, catalog: Optional[str], database: Optional[str], configuration: Mapping[str, str], libraries: Sequence[LibrariesGlob])` |
| `pipelines.cli.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `pipelines.cli.PySparkTypeError` | Class | `(...)` |
| `pipelines.cli.SparkConnectGraphElementRegistry` | Class | `(spark: SparkSession, dataflow_graph_id: str)` |
| `pipelines.cli.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pipelines.cli.add_pipeline_analysis_context` | Function | `(spark: SparkSession, dataflow_graph_id: str, flow_name: Optional[str]) -> Generator[None, None, None]` |
| `pipelines.cli.block_session_mutations` | Function | `() -> Generator[None, None, None]` |
| `pipelines.cli.change_dir` | Function | `(path: Path) -> Generator[None, None, None]` |
| `pipelines.cli.create_dataflow_graph` | Function | `(spark: SparkSession, default_catalog: Optional[str], default_database: Optional[str], sql_conf: Optional[Mapping[str, str]]) -> str` |
| `pipelines.cli.find_pipeline_spec` | Function | `(current_dir: Path) -> Path` |
| `pipelines.cli.graph_element_registration_context` | Function | `(registry: GraphElementRegistry) -> Generator[None, None, None]` |
| `pipelines.cli.handle_pipeline_events` | Function | `(iter: Iterator[Dict[str, Any]]) -> None` |
| `pipelines.cli.init` | Function | `(name: str) -> None` |
| `pipelines.cli.load_pipeline_spec` | Function | `(spec_path: Path) -> PipelineSpec` |
| `pipelines.cli.log_with_curr_timestamp` | Function | `(message: str) -> None` |
| `pipelines.cli.main` | Function | `() -> None` |
| `pipelines.cli.parse_table_list` | Function | `(value: str) -> List[str]` |
| `pipelines.cli.register_definitions` | Function | `(spec_path: Path, registry: GraphElementRegistry, spec: PipelineSpec, spark: SparkSession, dataflow_graph_id: str) -> None` |
| `pipelines.cli.run` | Function | `(spec_path: Path, full_refresh: Sequence[str], full_refresh_all: bool, refresh: Sequence[str], dry: bool) -> None` |
| `pipelines.cli.start_run` | Function | `(spark: SparkSession, dataflow_graph_id: str, full_refresh: Optional[Sequence[str]], full_refresh_all: bool, refresh: Optional[Sequence[str]], dry: bool, storage: str) -> Iterator[Dict[str, Any]]` |
| `pipelines.cli.unpack_pipeline_spec` | Function | `(spec_data: Mapping[str, Any]) -> PipelineSpec` |
| `pipelines.cli.validate_patch_glob_pattern` | Function | `(glob_pattern: str) -> str` |
| `pipelines.cli.validate_str_dict` | Function | `(d: Mapping[str, str], field_name: str) -> Mapping[str, str]` |
| `pipelines.create_sink` | Function | `(name: str, format: str, options: Optional[Dict[str, str]]) -> None` |
| `pipelines.create_streaming_table` | Function | `(name: str, comment: Optional[str], table_properties: Optional[Dict[str, str]], partition_cols: Optional[List[str]], cluster_by: Optional[List[str]], schema: Optional[Union[StructType, str]], format: Optional[str]) -> None` |
| `pipelines.flow.DataFrame` | Class | `(...)` |
| `pipelines.flow.Flow` | Class | `(name: str, target: str, spark_conf: Dict[str, str], source_code_location: SourceCodeLocation, func: QueryFunction)` |
| `pipelines.flow.QueryFunction` | Object | `` |
| `pipelines.flow.SourceCodeLocation` | Class | `(filename: str, line_number: Optional[int])` |
| `pipelines.graph_element_registry.Flow` | Class | `(name: str, target: str, spark_conf: Dict[str, str], source_code_location: SourceCodeLocation, func: QueryFunction)` |
| `pipelines.graph_element_registry.GraphElementRegistry` | Class | `(...)` |
| `pipelines.graph_element_registry.Output` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation)` |
| `pipelines.graph_element_registry.PySparkRuntimeError` | Class | `(...)` |
| `pipelines.graph_element_registry.get_active_graph_element_registry` | Function | `() -> GraphElementRegistry` |
| `pipelines.graph_element_registry.graph_element_registration_context` | Function | `(registry: GraphElementRegistry) -> Generator[None, None, None]` |
| `pipelines.init_cli.PYTHON_EXAMPLE` | Object | `` |
| `pipelines.init_cli.SPEC` | Object | `` |
| `pipelines.init_cli.SQL_EXAMPLE` | Object | `` |
| `pipelines.init_cli.init` | Function | `(name: str) -> None` |
| `pipelines.logging_utils.log_with_curr_timestamp` | Function | `(message: str) -> None` |
| `pipelines.logging_utils.log_with_provided_timestamp` | Function | `(message: str, timestamp: datetime) -> None` |
| `pipelines.materialized_view` | Function | `(query_function: Optional[QueryFunction], name: Optional[str], comment: Optional[str], spark_conf: Optional[Dict[str, str]], table_properties: Optional[Dict[str, str]], partition_cols: Optional[List[str]], cluster_by: Optional[List[str]], schema: Optional[Union[StructType, str]], format: Optional[str]) -> Union[Callable[[QueryFunction], None], None]` |
| `pipelines.output.MaterializedView` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.output.Output` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation)` |
| `pipelines.output.Sink` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, format: str, options: Mapping[str, str])` |
| `pipelines.output.SourceCodeLocation` | Class | `(filename: str, line_number: Optional[int])` |
| `pipelines.output.StreamingTable` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.output.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pipelines.output.Table` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.output.TemporaryView` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation)` |
| `pipelines.source_code_location.SourceCodeLocation` | Class | `(filename: str, line_number: Optional[int])` |
| `pipelines.source_code_location.get_caller_source_code_location` | Function | `(stacklevel: int) -> SourceCodeLocation` |
| `pipelines.spark_connect_graph_element_registry.ConnectDataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `pipelines.spark_connect_graph_element_registry.Flow` | Class | `(name: str, target: str, spark_conf: Dict[str, str], source_code_location: SourceCodeLocation, func: QueryFunction)` |
| `pipelines.spark_connect_graph_element_registry.GraphElementRegistry` | Class | `(...)` |
| `pipelines.spark_connect_graph_element_registry.MaterializedView` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.spark_connect_graph_element_registry.Output` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation)` |
| `pipelines.spark_connect_graph_element_registry.PySparkTypeError` | Class | `(...)` |
| `pipelines.spark_connect_graph_element_registry.Sink` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, format: str, options: Mapping[str, str])` |
| `pipelines.spark_connect_graph_element_registry.SourceCodeLocation` | Class | `(filename: str, line_number: Optional[int])` |
| `pipelines.spark_connect_graph_element_registry.SparkConnectGraphElementRegistry` | Class | `(spark: SparkSession, dataflow_graph_id: str)` |
| `pipelines.spark_connect_graph_element_registry.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pipelines.spark_connect_graph_element_registry.StreamingTable` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.spark_connect_graph_element_registry.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `pipelines.spark_connect_graph_element_registry.Table` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation, table_properties: Mapping[str, str], partition_cols: Optional[Sequence[str]], cluster_by: Optional[Sequence[str]], schema: Optional[Union[StructType, str]], format: Optional[str])` |
| `pipelines.spark_connect_graph_element_registry.TemporaryView` | Class | `(name: str, comment: Optional[str], source_code_location: SourceCodeLocation)` |
| `pipelines.spark_connect_graph_element_registry.add_pipeline_analysis_context` | Function | `(spark: SparkSession, dataflow_graph_id: str, flow_name: Optional[str]) -> Generator[None, None, None]` |
| `pipelines.spark_connect_graph_element_registry.block_spark_connect_execution_and_analysis` | Function | `() -> Generator[None, None, None]` |
| `pipelines.spark_connect_graph_element_registry.pb2` | Object | `` |
| `pipelines.spark_connect_graph_element_registry.pyspark_types_to_proto_types` | Function | `(data_type: DataType) -> pb2.DataType` |
| `pipelines.spark_connect_graph_element_registry.source_code_location_to_proto` | Function | `(source_code_location: SourceCodeLocation) -> pb2.SourceCodeLocation` |
| `pipelines.spark_connect_pipeline.PySparkValueError` | Class | `(...)` |
| `pipelines.spark_connect_pipeline.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `pipelines.spark_connect_pipeline.create_dataflow_graph` | Function | `(spark: SparkSession, default_catalog: Optional[str], default_database: Optional[str], sql_conf: Optional[Mapping[str, str]]) -> str` |
| `pipelines.spark_connect_pipeline.handle_pipeline_events` | Function | `(iter: Iterator[Dict[str, Any]]) -> None` |
| `pipelines.spark_connect_pipeline.log_with_provided_timestamp` | Function | `(message: str, timestamp: datetime) -> None` |
| `pipelines.spark_connect_pipeline.pb2` | Object | `` |
| `pipelines.spark_connect_pipeline.start_run` | Function | `(spark: SparkSession, dataflow_graph_id: str, full_refresh: Optional[Sequence[str]], full_refresh_all: bool, refresh: Optional[Sequence[str]], dry: bool, storage: str) -> Iterator[Dict[str, Any]]` |
| `pipelines.table` | Function | `(query_function: Optional[QueryFunction], name: Optional[str], comment: Optional[str], spark_conf: Optional[Dict[str, str]], table_properties: Optional[Dict[str, str]], partition_cols: Optional[List[str]], cluster_by: Optional[List[str]], schema: Optional[Union[StructType, str]], format: Optional[str]) -> Union[Callable[[QueryFunction], None], None]` |
| `pipelines.temporary_view` | Function | `(query_function: Optional[QueryFunction], name: Optional[str], comment: Optional[str], spark_conf: Optional[Dict[str, str]]) -> Union[Callable[[QueryFunction], None], None]` |
| `pipelines.type_error_utils.PySparkTypeError` | Class | `(...)` |
| `pipelines.type_error_utils.validate_optional_list_of_str_arg` | Function | `(arg_name: str, arg_value: Optional[List[str]]) -> None` |
| `profiler.AccumulatorParam` | Class | `(...)` |
| `profiler.BasicProfiler` | Class | `(ctx: SparkContext)` |
| `profiler.CodeMapDict` | Object | `` |
| `profiler.CodeMapForUDF` | Class | `(...)` |
| `profiler.CodeMapForUDFV2` | Class | `(...)` |
| `profiler.LineProfile` | Object | `` |
| `profiler.MemUsageParam` | Class | `(...)` |
| `profiler.MemoryProfiler` | Class | `(ctx: SparkContext)` |
| `profiler.MemoryTuple` | Object | `` |
| `profiler.PStatsParam` | Class | `(...)` |
| `profiler.Profiler` | Class | `(ctx: SparkContext)` |
| `profiler.ProfilerCollector` | Class | `(profiler_cls: Type[Profiler], udf_profiler_cls: Type[Profiler], memory_profiler_cls: Type[Profiler], dump_path: Optional[str])` |
| `profiler.PySparkRuntimeError` | Class | `(...)` |
| `profiler.PySparkValueError` | Class | `(...)` |
| `profiler.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `profiler.UDFBasicProfiler` | Class | `(...)` |
| `profiler.UDFLineProfiler` | Class | `(kw: Any)` |
| `profiler.UDFLineProfilerV2` | Class | `(kw: Any)` |
| `profiler.has_memory_profiler` | Object | `` |
| `rdd` | Object | `` |
| `rddsampler.RDDRangeSampler` | Class | `(lowerBound, upperBound, seed)` |
| `rddsampler.RDDSampler` | Class | `(withReplacement, fraction, seed)` |
| `rddsampler.RDDSamplerBase` | Class | `(withReplacement, seed)` |
| `rddsampler.RDDStratifiedSampler` | Class | `(withReplacement, fractions, seed)` |
| `resource.ExecutorResourceRequest` | Class | `(resourceName: str, amount: int, discoveryScript: str, vendor: str)` |
| `resource.ExecutorResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, ExecutorResourceRequest]])` |
| `resource.ResourceInformation` | Class | `(name: str, addresses: List[str])` |
| `resource.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `resource.ResourceProfileBuilder` | Class | `()` |
| `resource.TaskResourceRequest` | Class | `(resourceName: str, amount: float)` |
| `resource.TaskResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, TaskResourceRequest]])` |
| `resource.information.ResourceInformation` | Class | `(name: str, addresses: List[str])` |
| `resource.profile.ExecutorResourceRequest` | Class | `(resourceName: str, amount: int, discoveryScript: str, vendor: str)` |
| `resource.profile.ExecutorResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, ExecutorResourceRequest]])` |
| `resource.profile.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `resource.profile.ResourceProfileBuilder` | Class | `()` |
| `resource.profile.TaskResourceRequest` | Class | `(resourceName: str, amount: float)` |
| `resource.profile.TaskResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, TaskResourceRequest]])` |
| `resource.requests.ExecutorResourceRequest` | Class | `(resourceName: str, amount: int, discoveryScript: str, vendor: str)` |
| `resource.requests.ExecutorResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, ExecutorResourceRequest]])` |
| `resource.requests.TaskResourceRequest` | Class | `(resourceName: str, amount: float)` |
| `resource.requests.TaskResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, TaskResourceRequest]])` |
| `resultiterable.ResultIterable` | Class | `(data: SizedIterable[T])` |
| `resultiterable.SizedIterable` | Class | `(...)` |
| `resultiterable.T` | Object | `` |
| `serializers.AutoBatchedSerializer` | Class | `(serializer, bestSize)` |
| `serializers.AutoSerializer` | Class | `()` |
| `serializers.BatchedSerializer` | Class | `(serializer, batchSize)` |
| `serializers.CPickleSerializer` | Object | `` |
| `serializers.CartesianDeserializer` | Class | `(key_ser, val_ser)` |
| `serializers.ChunkedStream` | Class | `(wrapped, buffer_size)` |
| `serializers.CloudPickleSerializer` | Class | `(...)` |
| `serializers.CompressedSerializer` | Class | `(serializer)` |
| `serializers.FlattenedValuesSerializer` | Class | `(serializer, batchSize)` |
| `serializers.FramedSerializer` | Class | `(...)` |
| `serializers.MarshalSerializer` | Class | `(...)` |
| `serializers.NoOpSerializer` | Class | `(...)` |
| `serializers.PairDeserializer` | Class | `(key_ser, val_ser)` |
| `serializers.PickleSerializer` | Class | `(...)` |
| `serializers.Serializer` | Class | `(...)` |
| `serializers.SpecialLengths` | Class | `(...)` |
| `serializers.UTF8Deserializer` | Class | `(use_unicode)` |
| `serializers.cloudpickle` | Object | `` |
| `serializers.pack_long` | Function | `(value)` |
| `serializers.pickle_protocol` | Object | `` |
| `serializers.read_bool` | Function | `(stream)` |
| `serializers.read_int` | Function | `(stream)` |
| `serializers.read_long` | Function | `(stream)` |
| `serializers.write_int` | Function | `(value, stream)` |
| `serializers.write_long` | Function | `(value, stream)` |
| `serializers.write_with_length` | Function | `(obj, stream)` |
| `shell.PROGRESS_BAR_ENABLED` | Object | `` |
| `shell.SPARK_LOG_SCHEMA` | Object | `` |
| `shell.SQLContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jsqlContext: Optional[JavaObject])` |
| `shell.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `shell.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `shell.code` | Object | `` |
| `shell.is_remote` | Function | `() -> bool` |
| `shell.parent_dir` | Object | `` |
| `shell.sc` | Object | `` |
| `shell.spark` | Object | `` |
| `shell.sql` | Object | `` |
| `shell.sqlContext` | Object | `` |
| `shell.sqlCtx` | Object | `` |
| `shell.url` | Object | `` |
| `shell.val` | Object | `` |
| `shell.version` | Object | `` |
| `shuffle.Aggregator` | Class | `(createCombiner, mergeValue, mergeCombiners)` |
| `shuffle.AutoBatchedSerializer` | Class | `(serializer, bestSize)` |
| `shuffle.BatchedSerializer` | Class | `(serializer, batchSize)` |
| `shuffle.CPickleSerializer` | Object | `` |
| `shuffle.CompressedSerializer` | Class | `(serializer)` |
| `shuffle.DiskBytesSpilled` | Object | `` |
| `shuffle.ExternalGroupBy` | Class | `(...)` |
| `shuffle.ExternalList` | Class | `(values)` |
| `shuffle.ExternalListOfList` | Class | `(values)` |
| `shuffle.ExternalMerger` | Class | `(aggregator, memory_limit, serializer, localdirs, scale, partitions, batch)` |
| `shuffle.ExternalSorter` | Class | `(memory_limit, serializer)` |
| `shuffle.FlattenedValuesSerializer` | Class | `(serializer, batchSize)` |
| `shuffle.GroupByKey` | Class | `(iterator)` |
| `shuffle.MemoryBytesSpilled` | Object | `` |
| `shuffle.Merger` | Class | `(aggregator)` |
| `shuffle.SimpleAggregator` | Class | `(combiner)` |
| `shuffle.fail_on_stopiteration` | Function | `(f: Callable) -> Callable` |
| `shuffle.get_used_memory` | Function | `()` |
| `shuffle.process` | Object | `` |
| `since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `sql.Catalog` | Class | `(sparkSession: SparkSession)` |
| `sql.Column` | Class | `(...)` |
| `sql.DataFrame` | Class | `(...)` |
| `sql.DataFrameNaFunctions` | Class | `(df: DataFrame)` |
| `sql.DataFrameReader` | Class | `(spark: SparkSession)` |
| `sql.DataFrameStatFunctions` | Class | `(df: DataFrame)` |
| `sql.DataFrameWriter` | Class | `(df: DataFrame)` |
| `sql.DataFrameWriterV2` | Class | `(df: DataFrame, table: str)` |
| `sql.Geography` | Class | `(wkb: bytes, srid: int)` |
| `sql.Geometry` | Class | `(wkb: bytes, srid: int)` |
| `sql.GroupedData` | Class | `(jgd: JavaObject, df: DataFrame)` |
| `sql.HiveContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jhiveContext: Optional[JavaObject])` |
| `sql.MergeIntoWriter` | Class | `(df: DataFrame, table: str, condition: Column)` |
| `sql.Observation` | Class | `(name: Optional[str])` |
| `sql.PandasCogroupedOps` | Class | `(gd1: GroupedData, gd2: GroupedData)` |
| `sql.Row` | Class | `(...)` |
| `sql.SQLContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jsqlContext: Optional[JavaObject])` |
| `sql.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.UDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.UDTFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.VariantVal` | Class | `(value: bytes, metadata: bytes)` |
| `sql.Window` | Class | `(...)` |
| `sql.WindowSpec` | Class | `(...)` |
| `sql.avro.functions.Column` | Class | `(...)` |
| `sql.avro.functions.ColumnOrName` | Object | `` |
| `sql.avro.functions.PySparkTypeError` | Class | `(...)` |
| `sql.avro.functions.from_avro` | Function | `(data: ColumnOrName, jsonFormatSchema: str, options: Optional[Dict[str, str]]) -> Column` |
| `sql.avro.functions.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.avro.functions.to_avro` | Function | `(data: ColumnOrName, jsonFormatSchema: str) -> Column` |
| `sql.avro.functions.try_remote_avro_functions` | Function | `(f: FuncT) -> FuncT` |
| `sql.catalog.Catalog` | Class | `(sparkSession: SparkSession)` |
| `sql.catalog.CatalogMetadata` | Class | `(...)` |
| `sql.catalog.Column` | Class | `(...)` |
| `sql.catalog.DataFrame` | Class | `(...)` |
| `sql.catalog.DataTypeOrString` | Object | `` |
| `sql.catalog.Database` | Class | `(...)` |
| `sql.catalog.Function` | Class | `(...)` |
| `sql.catalog.PySparkTypeError` | Class | `(...)` |
| `sql.catalog.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.catalog.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.catalog.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.catalog.Table` | Class | `(...)` |
| `sql.catalog.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.classic.column.Column` | Class | `(jc: JavaObject)` |
| `sql.classic.column.ColumnOrName` | Object | `` |
| `sql.classic.column.DataType` | Class | `(...)` |
| `sql.classic.column.DateTimeLiteral` | Object | `` |
| `sql.classic.column.DecimalLiteral` | Object | `` |
| `sql.classic.column.LiteralType` | Object | `` |
| `sql.classic.column.ParentColumn` | Class | `(...)` |
| `sql.classic.column.PySparkAttributeError` | Class | `(...)` |
| `sql.classic.column.PySparkTypeError` | Class | `(...)` |
| `sql.classic.column.PySparkValueError` | Class | `(...)` |
| `sql.classic.column.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.classic.column.WindowSpec` | Class | `(...)` |
| `sql.classic.column.enum_to_value` | Function | `(value: Any) -> Any` |
| `sql.classic.column.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.classic.column.with_origin_to_class` | Function | `(cls_or_ignores: Optional[Union[Type[T], List[str]]], ignores: Optional[List[str]]) -> Union[Type[T], Callable[[Type[T]], Type[T]]]` |
| `sql.classic.dataframe.AnalysisException` | Class | `(...)` |
| `sql.classic.dataframe.ArrowMapIterFunction` | Object | `` |
| `sql.classic.dataframe.BatchedSerializer` | Class | `(serializer, batchSize)` |
| `sql.classic.dataframe.CPickleSerializer` | Object | `` |
| `sql.classic.dataframe.Column` | Class | `(...)` |
| `sql.classic.dataframe.ColumnOrName` | Object | `` |
| `sql.classic.dataframe.ColumnOrNameOrOrdinal` | Object | `` |
| `sql.classic.dataframe.DataFrame` | Class | `(jdf: JavaObject, sql_ctx: Union[SQLContext, SparkSession])` |
| `sql.classic.dataframe.DataFrameNaFunctions` | Class | `(df: ParentDataFrame)` |
| `sql.classic.dataframe.DataFrameStatFunctions` | Class | `(df: ParentDataFrame)` |
| `sql.classic.dataframe.DataFrameWriter` | Class | `(df: DataFrame)` |
| `sql.classic.dataframe.DataFrameWriterV2` | Class | `(df: DataFrame, table: str)` |
| `sql.classic.dataframe.DataStreamWriter` | Class | `(df: DataFrame)` |
| `sql.classic.dataframe.ExecutionInfo` | Class | `(metrics: Optional[list[PlanMetrics]], obs: Optional[Sequence[ObservedMetrics]])` |
| `sql.classic.dataframe.F` | Object | `` |
| `sql.classic.dataframe.GroupedData` | Class | `(jgd: JavaObject, df: DataFrame)` |
| `sql.classic.dataframe.LiteralType` | Object | `` |
| `sql.classic.dataframe.MergeIntoWriter` | Class | `(df: DataFrame, table: str, condition: Column)` |
| `sql.classic.dataframe.Observation` | Class | `(name: Optional[str])` |
| `sql.classic.dataframe.OptionalPrimitiveType` | Object | `` |
| `sql.classic.dataframe.PandasConversionMixin` | Class | `(...)` |
| `sql.classic.dataframe.PandasDataFrameLike` | Object | `` |
| `sql.classic.dataframe.PandasMapIterFunction` | Object | `` |
| `sql.classic.dataframe.PandasMapOpsMixin` | Class | `(...)` |
| `sql.classic.dataframe.PandasOnSparkDataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `sql.classic.dataframe.ParentDataFrame` | Class | `(...)` |
| `sql.classic.dataframe.ParentDataFrameNaFunctions` | Class | `(df: DataFrame)` |
| `sql.classic.dataframe.ParentDataFrameStatFunctions` | Class | `(df: DataFrame)` |
| `sql.classic.dataframe.PrimitiveType` | Object | `` |
| `sql.classic.dataframe.PySparkAttributeError` | Class | `(...)` |
| `sql.classic.dataframe.PySparkIndexError` | Class | `(...)` |
| `sql.classic.dataframe.PySparkTypeError` | Class | `(...)` |
| `sql.classic.dataframe.PySparkValueError` | Class | `(...)` |
| `sql.classic.dataframe.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `sql.classic.dataframe.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `sql.classic.dataframe.Row` | Class | `(...)` |
| `sql.classic.dataframe.SCCallSiteSync` | Class | `(sc)` |
| `sql.classic.dataframe.SQLContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jsqlContext: Optional[JavaObject])` |
| `sql.classic.dataframe.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.classic.dataframe.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.classic.dataframe.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.classic.dataframe.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.classic.dataframe.TableArg` | Class | `(...)` |
| `sql.classic.dataframe.UTF8Deserializer` | Class | `(use_unicode)` |
| `sql.classic.dataframe.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.classic.dataframe.to_java_array` | Function | `(gateway: JavaGateway, jtype: JavaClass, arr: Sequence[Any]) -> JavaArray` |
| `sql.classic.dataframe.to_scala_map` | Function | `(jvm: JVMView, dic: Dict) -> JavaObject` |
| `sql.classic.table_arg.ColumnOrName` | Object | `` |
| `sql.classic.table_arg.ParentTableArg` | Class | `(...)` |
| `sql.classic.table_arg.TableArg` | Class | `(j_table_arg: JavaObject)` |
| `sql.classic.table_arg.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.classic.window.ColumnOrName` | Object | `` |
| `sql.classic.window.ParentWindow` | Class | `(...)` |
| `sql.classic.window.ParentWindowSpec` | Class | `(...)` |
| `sql.classic.window.Window` | Class | `(...)` |
| `sql.classic.window.WindowSpec` | Class | `(jspec: JavaObject)` |
| `sql.classic.window.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.column.Column` | Class | `(...)` |
| `sql.column.DataType` | Class | `(...)` |
| `sql.column.DateTimeLiteral` | Object | `` |
| `sql.column.DecimalLiteral` | Object | `` |
| `sql.column.LiteralType` | Object | `` |
| `sql.column.PySparkValueError` | Class | `(...)` |
| `sql.column.TableValuedFunctionArgument` | Class | `(...)` |
| `sql.column.WindowSpec` | Class | `(...)` |
| `sql.column.dispatch_col_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.conf.PySparkTypeError` | Class | `(...)` |
| `sql.conf.RuntimeConfig` | Class | `(jconf: JavaObject)` |
| `sql.connect.avro.functions.Column` | Class | `(...)` |
| `sql.connect.avro.functions.ColumnOrName` | Object | `` |
| `sql.connect.avro.functions.PyAvroFunctions` | Object | `` |
| `sql.connect.avro.functions.PySparkTypeError` | Class | `(...)` |
| `sql.connect.avro.functions.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.avro.functions.from_avro` | Function | `(data: ColumnOrName, jsonFormatSchema: str, options: Optional[Dict[str, str]]) -> Column` |
| `sql.connect.avro.functions.lit` | Function | `(col: Any) -> Column` |
| `sql.connect.avro.functions.to_avro` | Function | `(data: ColumnOrName, jsonFormatSchema: str) -> Column` |
| `sql.connect.catalog.Catalog` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.catalog.CatalogMetadata` | Class | `(...)` |
| `sql.connect.catalog.Column` | Class | `(...)` |
| `sql.connect.catalog.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.catalog.DataTypeOrString` | Object | `` |
| `sql.connect.catalog.Database` | Class | `(...)` |
| `sql.connect.catalog.Function` | Class | `(...)` |
| `sql.connect.catalog.PySparkCatalog` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.catalog.PySparkTypeError` | Class | `(...)` |
| `sql.connect.catalog.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.catalog.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.connect.catalog.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.catalog.Table` | Class | `(...)` |
| `sql.connect.catalog.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.connect.catalog.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.catalog.plan` | Object | `` |
| `sql.connect.client.ChannelBuilder` | Class | `(channelOptions: Optional[List[Tuple[str, Any]]], params: Optional[Dict[str, str]])` |
| `sql.connect.client.DefaultChannelBuilder` | Class | `(url: str, channelOptions: Optional[List[Tuple[str, Any]]])` |
| `sql.connect.client.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.client.artifact.ARCHIVE_PREFIX` | Object | `` |
| `sql.connect.client.artifact.Artifact` | Class | `(path: str, storage: LocalData)` |
| `sql.connect.client.artifact.ArtifactManager` | Class | `(user_id: Optional[str], session_id: str, channel: grpc.Channel, metadata: Iterable[Tuple[str, str]])` |
| `sql.connect.client.artifact.CACHE_PREFIX` | Object | `` |
| `sql.connect.client.artifact.FILE_PREFIX` | Object | `` |
| `sql.connect.client.artifact.FORWARD_TO_FS_PREFIX` | Object | `` |
| `sql.connect.client.artifact.InMemory` | Class | `(blob: bytes)` |
| `sql.connect.client.artifact.JAR_PREFIX` | Object | `` |
| `sql.connect.client.artifact.LocalData` | Class | `(...)` |
| `sql.connect.client.artifact.LocalFile` | Class | `(path: str)` |
| `sql.connect.client.artifact.PYFILE_PREFIX` | Object | `` |
| `sql.connect.client.artifact.PySparkRuntimeError` | Class | `(...)` |
| `sql.connect.client.artifact.PySparkValueError` | Class | `(...)` |
| `sql.connect.client.artifact.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.client.artifact.grpc_lib` | Object | `` |
| `sql.connect.client.artifact.logger` | Object | `` |
| `sql.connect.client.artifact.new_archive_artifact` | Function | `(file_name: str, storage: LocalData) -> Artifact` |
| `sql.connect.client.artifact.new_cache_artifact` | Function | `(id: str, storage: LocalData) -> Artifact` |
| `sql.connect.client.artifact.new_file_artifact` | Function | `(file_name: str, storage: LocalData) -> Artifact` |
| `sql.connect.client.artifact.new_jar_artifact` | Function | `(file_name: str, storage: LocalData) -> Artifact` |
| `sql.connect.client.artifact.new_pyfile_artifact` | Function | `(file_name: str, storage: LocalData) -> Artifact` |
| `sql.connect.client.artifact.proto` | Object | `` |
| `sql.connect.client.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.client.core.AnalyzeResult` | Class | `(schema: Optional[DataType], explain_string: Optional[str], tree_string: Optional[str], is_local: Optional[bool], is_streaming: Optional[bool], input_files: Optional[List[str]], spark_version: Optional[str], parsed: Optional[DataType], is_same_semantics: Optional[bool], semantic_hash: Optional[int], storage_level: Optional[StorageLevel], ddl_string: Optional[str])` |
| `sql.connect.client.core.ArtifactManager` | Class | `(user_id: Optional[str], session_id: str, channel: grpc.Channel, metadata: Iterable[Tuple[str, str]])` |
| `sql.connect.client.core.ChannelBuilder` | Class | `(channelOptions: Optional[List[Tuple[str, Any]]], params: Optional[Dict[str, str]])` |
| `sql.connect.client.core.CommonInlineUserDefinedDataSource` | Class | `(name: str, data_source: PythonDataSource)` |
| `sql.connect.client.core.CommonInlineUserDefinedFunction` | Class | `(function_name: str, function: Union[PythonUDF, JavaUDF], deterministic: bool, arguments: Optional[Sequence[Expression]])` |
| `sql.connect.client.core.CommonInlineUserDefinedTableFunction` | Class | `(function_name: str, function: PythonUDTF, deterministic: bool, arguments: Sequence[Expression])` |
| `sql.connect.client.core.ConfigResult` | Class | `(pairs: List[Tuple[str, Optional[str]]], warnings: List[str])` |
| `sql.connect.client.core.ConnectProfilerCollector` | Class | `()` |
| `sql.connect.client.core.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.connect.client.core.DataType` | Class | `(...)` |
| `sql.connect.client.core.DataTypeOrString` | Object | `` |
| `sql.connect.client.core.DefaultChannelBuilder` | Class | `(url: str, channelOptions: Optional[List[Tuple[str, Any]]])` |
| `sql.connect.client.core.DefaultPolicy` | Class | `(max_retries: Optional[int], backoff_multiplier: float, initial_backoff: int, max_backoff: Optional[int], jitter: int, min_jitter_threshold: int, recognize_server_retry_delay: bool, max_server_retry_delay: Optional[int])` |
| `sql.connect.client.core.ExecutePlanResponseReattachableIterator` | Class | `(request: pb2.ExecutePlanRequest, stub: grpc_lib.SparkConnectServiceStub, retrying: Callable[[], Retrying], metadata: Iterable[Tuple[str, str]])` |
| `sql.connect.client.core.ExecutionInfo` | Class | `(metrics: Optional[list[PlanMetrics]], obs: Optional[Sequence[ObservedMetrics]])` |
| `sql.connect.client.core.JavaUDF` | Class | `(class_name: str, output_type: Optional[Union[DataType, str]], aggregate: bool)` |
| `sql.connect.client.core.LiteralExpression` | Class | `(value: Any, dataType: DataType)` |
| `sql.connect.client.core.MetricValue` | Class | `(name: str, value: Union[int, float], type: str)` |
| `sql.connect.client.core.Observation` | Class | `(name: Optional[str])` |
| `sql.connect.client.core.ObservedMetrics` | Class | `(...)` |
| `sql.connect.client.core.PlanMetrics` | Class | `(name: str, id: int, parent: int, metrics: List[MetricValue])` |
| `sql.connect.client.core.PlanObservedMetrics` | Class | `(name: str, metrics: List[pb2.Expression.Literal], keys: List[str])` |
| `sql.connect.client.core.Progress` | Class | `(char: str, min_width: int, output: typing.IO, enabled: bool, handlers: Iterable[ProgressHandler], operation_id: typing.Optional[str])` |
| `sql.connect.client.core.ProgressHandler` | Class | `(...)` |
| `sql.connect.client.core.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.connect.client.core.PySparkNotImplementedError` | Class | `(...)` |
| `sql.connect.client.core.PySparkValueError` | Class | `(...)` |
| `sql.connect.client.core.PythonDataSource` | Class | `(data_source: Type, python_ver: str)` |
| `sql.connect.client.core.PythonEvalType` | Class | `(...)` |
| `sql.connect.client.core.PythonUDF` | Class | `(output_type: Union[DataType, str], eval_type: int, func: Callable[..., Any], python_ver: str)` |
| `sql.connect.client.core.PythonUDTF` | Class | `(func: Type, return_type: Optional[Union[DataType, str]], eval_type: int, python_ver: str)` |
| `sql.connect.client.core.ResourceInformation` | Class | `(name: str, addresses: List[str])` |
| `sql.connect.client.core.RetryPolicy` | Class | `(max_retries: Optional[int], initial_backoff: int, max_backoff: Optional[int], backoff_multiplier: float, jitter: int, min_jitter_threshold: int, recognize_server_retry_delay: bool, max_server_retry_delay: Optional[int])` |
| `sql.connect.client.core.Retrying` | Class | `(policies: typing.Union[RetryPolicy, typing.Iterable[RetryPolicy]], sleep: Callable[[float], None])` |
| `sql.connect.client.core.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.client.core.SparkConnectException` | Class | `(...)` |
| `sql.connect.client.core.SparkConnectGrpcException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], reason: Optional[str], sql_state: Optional[str], server_stacktrace: Optional[str], display_server_stacktrace: bool, contexts: Optional[List[BaseQueryContext]], grpc_status_code: grpc.StatusCode, breaking_change_info: Optional[Dict[str, Any]])` |
| `sql.connect.client.core.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.client.core.SpecialAccumulatorIds` | Class | `(...)` |
| `sql.connect.client.core.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.connect.client.core.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.client.core.TimestampType` | Class | `(...)` |
| `sql.connect.client.core.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.client.core.convert_exception` | Function | `(info: ErrorInfo, truncated_message: str, resp: Optional[pb2.FetchErrorDetailsResponse], display_server_stacktrace: bool, grpc_status_code: grpc.StatusCode) -> SparkConnectException` |
| `sql.connect.client.core.from_arrow_schema` | Function | `(arrow_schema: pa.Schema, prefer_timestamp_ntz: bool) -> StructType` |
| `sql.connect.client.core.from_proto` | Function | `(proto: ExecutePlanResponse.ExecutionProgress) -> typing.Tuple[Iterable[StageInfo], int]` |
| `sql.connect.client.core.get_python_ver` | Function | `() -> str` |
| `sql.connect.client.core.grpc_lib` | Object | `` |
| `sql.connect.client.core.is_remote_only` | Function | `() -> bool` |
| `sql.connect.client.core.logger` | Object | `` |
| `sql.connect.client.core.pb2` | Object | `` |
| `sql.connect.client.core.proto_to_remote_cached_dataframe` | Function | `(relation: pb2.CachedRemoteRelation) -> DataFrame` |
| `sql.connect.client.core.proto_to_storage_level` | Function | `(storage_level: pb2.StorageLevel) -> StorageLevel` |
| `sql.connect.client.core.storage_level_to_proto` | Function | `(storage_level: StorageLevel) -> pb2.StorageLevel` |
| `sql.connect.client.core.types` | Object | `` |
| `sql.connect.client.getLogLevel` | Function | `() -> Optional[int]` |
| `sql.connect.client.reattach.ExecutePlanResponseReattachableIterator` | Class | `(request: pb2.ExecutePlanRequest, stub: grpc_lib.SparkConnectServiceStub, retrying: Callable[[], Retrying], metadata: Iterable[Tuple[str, str]])` |
| `sql.connect.client.reattach.PySparkRuntimeError` | Class | `(...)` |
| `sql.connect.client.reattach.RetryException` | Class | `(...)` |
| `sql.connect.client.reattach.Retrying` | Class | `(policies: typing.Union[RetryPolicy, typing.Iterable[RetryPolicy]], sleep: Callable[[float], None])` |
| `sql.connect.client.reattach.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.client.reattach.grpc_lib` | Object | `` |
| `sql.connect.client.reattach.logger` | Object | `` |
| `sql.connect.client.reattach.pb2` | Object | `` |
| `sql.connect.client.retries.AttemptManager` | Class | `(retrying: Retrying)` |
| `sql.connect.client.retries.DefaultPolicy` | Class | `(max_retries: Optional[int], backoff_multiplier: float, initial_backoff: int, max_backoff: Optional[int], jitter: int, min_jitter_threshold: int, recognize_server_retry_delay: bool, max_server_retry_delay: Optional[int])` |
| `sql.connect.client.retries.PySparkRuntimeError` | Class | `(...)` |
| `sql.connect.client.retries.RetryException` | Class | `(...)` |
| `sql.connect.client.retries.RetryPolicy` | Class | `(max_retries: Optional[int], initial_backoff: int, max_backoff: Optional[int], backoff_multiplier: float, jitter: int, min_jitter_threshold: int, recognize_server_retry_delay: bool, max_server_retry_delay: Optional[int])` |
| `sql.connect.client.retries.RetryPolicyState` | Class | `(policy: RetryPolicy)` |
| `sql.connect.client.retries.Retrying` | Class | `(policies: typing.Union[RetryPolicy, typing.Iterable[RetryPolicy]], sleep: Callable[[float], None])` |
| `sql.connect.client.retries.extract_retry_delay` | Function | `(exception: BaseException) -> Optional[int]` |
| `sql.connect.client.retries.extract_retry_info` | Function | `(exception: BaseException) -> Optional[error_details_pb2.RetryInfo]` |
| `sql.connect.client.retries.logger` | Object | `` |
| `sql.connect.column.CaseWhen` | Class | `(branches: Sequence[Tuple[Expression, Expression]], else_value: Optional[Expression])` |
| `sql.connect.column.CastExpression` | Class | `(expr: Expression, data_type: Union[DataType, str], eval_mode: Optional[str])` |
| `sql.connect.column.Column` | Class | `(expr: Expression)` |
| `sql.connect.column.DataType` | Class | `(...)` |
| `sql.connect.column.DateTimeLiteral` | Object | `` |
| `sql.connect.column.DecimalLiteral` | Object | `` |
| `sql.connect.column.DropField` | Class | `(structExpr: Expression, fieldName: str)` |
| `sql.connect.column.Expression` | Class | `()` |
| `sql.connect.column.LiteralExpression` | Class | `(value: Any, dataType: DataType)` |
| `sql.connect.column.LiteralType` | Object | `` |
| `sql.connect.column.ParentColumn` | Class | `(...)` |
| `sql.connect.column.PySparkAttributeError` | Class | `(...)` |
| `sql.connect.column.PySparkTypeError` | Class | `(...)` |
| `sql.connect.column.PySparkValueError` | Class | `(...)` |
| `sql.connect.column.SortOrder` | Class | `(child: Expression, ascending: bool, nullsFirst: bool)` |
| `sql.connect.column.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.column.SubqueryExpression` | Class | `(plan: LogicalPlan, subquery_type: str, partition_spec: Optional[Sequence[Expression]], order_spec: Optional[Sequence[SortOrder]], with_single_partition: Optional[bool], in_subquery_values: Optional[Sequence[Expression]])` |
| `sql.connect.column.UnresolvedExtractValue` | Class | `(child: Expression, extraction: Expression)` |
| `sql.connect.column.UnresolvedFunction` | Class | `(name: str, args: Sequence[Expression], is_distinct: bool)` |
| `sql.connect.column.WindowExpression` | Class | `(windowFunction: Expression, windowSpec: WindowSpec)` |
| `sql.connect.column.WindowSpec` | Class | `(partitionSpec: Sequence[Expression], orderSpec: Sequence[SortOrder], frame: Optional[WindowFrame])` |
| `sql.connect.column.WithField` | Class | `(structExpr: Expression, fieldName: str, valueExpr: Expression)` |
| `sql.connect.column.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.column.enum_to_value` | Function | `(value: Any) -> Any` |
| `sql.connect.column.proto` | Object | `` |
| `sql.connect.column.with_origin_to_class` | Function | `(cls_or_ignores: Optional[Union[Type[T], List[str]]], ignores: Optional[List[str]]) -> Union[Type[T], Callable[[Type[T]], Type[T]]]` |
| `sql.connect.conf.PySparkRuntimeConfig` | Class | `(jconf: JavaObject)` |
| `sql.connect.conf.PySparkTypeError` | Class | `(...)` |
| `sql.connect.conf.PySparkValueError` | Class | `(...)` |
| `sql.connect.conf.RuntimeConf` | Class | `(client: SparkConnectClient)` |
| `sql.connect.conf.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.conf.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.conf.proto` | Object | `` |
| `sql.connect.conversion.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.conversion.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.connect.conversion.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.conversion.pb2` | Object | `` |
| `sql.connect.conversion.proto_to_remote_cached_dataframe` | Function | `(relation: pb2.CachedRemoteRelation) -> DataFrame` |
| `sql.connect.conversion.proto_to_storage_level` | Function | `(storage_level: pb2.StorageLevel) -> StorageLevel` |
| `sql.connect.conversion.storage_level_to_proto` | Function | `(storage_level: StorageLevel) -> pb2.StorageLevel` |
| `sql.connect.dataframe.ArrowMapIterFunction` | Object | `` |
| `sql.connect.dataframe.ArrowTableToRowsConversion` | Class | `(...)` |
| `sql.connect.dataframe.CPickleSerializer` | Object | `` |
| `sql.connect.dataframe.Column` | Class | `(...)` |
| `sql.connect.dataframe.ColumnOrName` | Object | `` |
| `sql.connect.dataframe.ColumnOrNameOrOrdinal` | Object | `` |
| `sql.connect.dataframe.ColumnReference` | Class | `(unparsed_identifier: str, plan_id: Optional[int], is_metadata_column: bool)` |
| `sql.connect.dataframe.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.dataframe.DataFrameNaFunctions` | Class | `(df: ParentDataFrame)` |
| `sql.connect.dataframe.DataFrameStatFunctions` | Class | `(df: ParentDataFrame)` |
| `sql.connect.dataframe.DataFrameWriter` | Class | `(plan: LogicalPlan, session: SparkSession, callback: Optional[Callable[[ExecutionInfo], None]])` |
| `sql.connect.dataframe.DataFrameWriterV2` | Class | `(plan: LogicalPlan, session: SparkSession, table: str, callback: Optional[Callable[[ExecutionInfo], None]])` |
| `sql.connect.dataframe.DataStreamWriter` | Class | `(plan: LogicalPlan, session: SparkSession)` |
| `sql.connect.dataframe.DirectShufflePartitionID` | Class | `(child: Expression)` |
| `sql.connect.dataframe.ExecutionInfo` | Class | `(metrics: Optional[list[PlanMetrics]], obs: Optional[Sequence[ObservedMetrics]])` |
| `sql.connect.dataframe.F` | Object | `` |
| `sql.connect.dataframe.GroupedData` | Class | `(df: DataFrame, group_type: str, grouping_cols: Sequence[Column], pivot_col: Optional[Column], pivot_values: Optional[Sequence[LiteralType]], grouping_sets: Optional[Sequence[Sequence[Column]]])` |
| `sql.connect.dataframe.LiteralType` | Object | `` |
| `sql.connect.dataframe.MergeIntoWriter` | Class | `(plan: LogicalPlan, session: SparkSession, table: str, condition: Column, callback: Optional[Callable[[ExecutionInfo], None]])` |
| `sql.connect.dataframe.Observation` | Class | `(name: Optional[str])` |
| `sql.connect.dataframe.OptionalPrimitiveType` | Object | `` |
| `sql.connect.dataframe.PandasDataFrameLike` | Object | `` |
| `sql.connect.dataframe.PandasMapIterFunction` | Object | `` |
| `sql.connect.dataframe.PandasOnSparkDataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `sql.connect.dataframe.ParentDataFrame` | Class | `(...)` |
| `sql.connect.dataframe.ParentDataFrameNaFunctions` | Class | `(df: DataFrame)` |
| `sql.connect.dataframe.ParentDataFrameStatFunctions` | Class | `(df: DataFrame)` |
| `sql.connect.dataframe.PrimitiveType` | Object | `` |
| `sql.connect.dataframe.PySparkAttributeError` | Class | `(...)` |
| `sql.connect.dataframe.PySparkIndexError` | Class | `(...)` |
| `sql.connect.dataframe.PySparkNotImplementedError` | Class | `(...)` |
| `sql.connect.dataframe.PySparkRuntimeError` | Class | `(...)` |
| `sql.connect.dataframe.PySparkTypeError` | Class | `(...)` |
| `sql.connect.dataframe.PySparkValueError` | Class | `(...)` |
| `sql.connect.dataframe.PythonEvalType` | Class | `(...)` |
| `sql.connect.dataframe.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `sql.connect.dataframe.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `sql.connect.dataframe.Row` | Class | `(...)` |
| `sql.connect.dataframe.SessionNotSameException` | Class | `(...)` |
| `sql.connect.dataframe.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.dataframe.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.connect.dataframe.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.dataframe.SubqueryExpression` | Class | `(plan: LogicalPlan, subquery_type: str, partition_spec: Optional[Sequence[Expression]], order_spec: Optional[Sequence[SortOrder]], with_single_partition: Optional[bool], in_subquery_values: Optional[Sequence[Expression]])` |
| `sql.connect.dataframe.TableArg` | Class | `(...)` |
| `sql.connect.dataframe.UnresolvedRegex` | Class | `(col_name: str, plan_id: Optional[int])` |
| `sql.connect.dataframe.UnresolvedStar` | Class | `(unparsed_target: Optional[str], plan_id: Optional[int])` |
| `sql.connect.dataframe.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.dataframe.from_arrow_schema` | Function | `(arrow_schema: pa.Schema, prefer_timestamp_ntz: bool) -> StructType` |
| `sql.connect.dataframe.is_remote_only` | Function | `() -> bool` |
| `sql.connect.dataframe.logger` | Object | `` |
| `sql.connect.dataframe.plan` | Object | `` |
| `sql.connect.dataframe.to_arrow_schema` | Function | `(schema: StructType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.Schema` |
| `sql.connect.datasource.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.connect.datasource.DataSourceRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.datasource.PySparkDataSourceRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.datasource.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.datasource.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.expressions.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.connect.expressions.BinaryType` | Class | `(...)` |
| `sql.connect.expressions.BooleanType` | Class | `(...)` |
| `sql.connect.expressions.ByteType` | Class | `(...)` |
| `sql.connect.expressions.CallFunction` | Class | `(name: str, args: Sequence[Expression])` |
| `sql.connect.expressions.CaseWhen` | Class | `(branches: Sequence[Tuple[Expression, Expression]], else_value: Optional[Expression])` |
| `sql.connect.expressions.CastExpression` | Class | `(expr: Expression, data_type: Union[DataType, str], eval_mode: Optional[str])` |
| `sql.connect.expressions.CloudPickleSerializer` | Class | `(...)` |
| `sql.connect.expressions.ColumnAlias` | Class | `(child: Expression, alias: Sequence[str], metadata: Any)` |
| `sql.connect.expressions.ColumnReference` | Class | `(unparsed_identifier: str, plan_id: Optional[int], is_metadata_column: bool)` |
| `sql.connect.expressions.CommonInlineUserDefinedFunction` | Class | `(function_name: str, function: Union[PythonUDF, JavaUDF], deterministic: bool, arguments: Optional[Sequence[Expression]])` |
| `sql.connect.expressions.DataType` | Class | `(...)` |
| `sql.connect.expressions.DateType` | Class | `(...)` |
| `sql.connect.expressions.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `sql.connect.expressions.DecimalType` | Class | `(precision: int, scale: int)` |
| `sql.connect.expressions.DirectShufflePartitionID` | Class | `(child: Expression)` |
| `sql.connect.expressions.DistributedSequenceID` | Class | `(...)` |
| `sql.connect.expressions.DoubleType` | Class | `(...)` |
| `sql.connect.expressions.DropField` | Class | `(structExpr: Expression, fieldName: str)` |
| `sql.connect.expressions.Expression` | Class | `()` |
| `sql.connect.expressions.FloatType` | Class | `(...)` |
| `sql.connect.expressions.IntegerType` | Class | `(...)` |
| `sql.connect.expressions.JVM_BYTE_MAX` | Object | `` |
| `sql.connect.expressions.JVM_BYTE_MIN` | Object | `` |
| `sql.connect.expressions.JVM_INT_MAX` | Object | `` |
| `sql.connect.expressions.JVM_INT_MIN` | Object | `` |
| `sql.connect.expressions.JVM_LONG_MAX` | Object | `` |
| `sql.connect.expressions.JVM_LONG_MIN` | Object | `` |
| `sql.connect.expressions.JVM_SHORT_MAX` | Object | `` |
| `sql.connect.expressions.JVM_SHORT_MIN` | Object | `` |
| `sql.connect.expressions.JavaUDF` | Class | `(class_name: str, output_type: Optional[Union[DataType, str]], aggregate: bool)` |
| `sql.connect.expressions.LambdaFunction` | Class | `(function: Expression, arguments: Sequence[UnresolvedNamedLambdaVariable])` |
| `sql.connect.expressions.LiteralExpression` | Class | `(value: Any, dataType: DataType)` |
| `sql.connect.expressions.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `sql.connect.expressions.LongType` | Class | `(...)` |
| `sql.connect.expressions.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `sql.connect.expressions.NamedArgumentExpression` | Class | `(key: str, value: Expression)` |
| `sql.connect.expressions.NullType` | Class | `(...)` |
| `sql.connect.expressions.PySparkTypeError` | Class | `(...)` |
| `sql.connect.expressions.PySparkValueError` | Class | `(...)` |
| `sql.connect.expressions.PythonUDF` | Class | `(output_type: Union[DataType, str], eval_type: int, func: Callable[..., Any], python_ver: str)` |
| `sql.connect.expressions.SQLExpression` | Class | `(expr: str)` |
| `sql.connect.expressions.ShortType` | Class | `(...)` |
| `sql.connect.expressions.SortOrder` | Class | `(child: Expression, ascending: bool, nullsFirst: bool)` |
| `sql.connect.expressions.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.expressions.StringType` | Class | `(collation: str)` |
| `sql.connect.expressions.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.expressions.SubqueryExpression` | Class | `(plan: LogicalPlan, subquery_type: str, partition_spec: Optional[Sequence[Expression]], order_spec: Optional[Sequence[SortOrder]], with_single_partition: Optional[bool], in_subquery_values: Optional[Sequence[Expression]])` |
| `sql.connect.expressions.TimeType` | Class | `(precision: int)` |
| `sql.connect.expressions.TimestampNTZType` | Class | `(...)` |
| `sql.connect.expressions.TimestampType` | Class | `(...)` |
| `sql.connect.expressions.UnparsedDataType` | Class | `(data_type_string: str)` |
| `sql.connect.expressions.UnresolvedExtractValue` | Class | `(child: Expression, extraction: Expression)` |
| `sql.connect.expressions.UnresolvedFunction` | Class | `(name: str, args: Sequence[Expression], is_distinct: bool)` |
| `sql.connect.expressions.UnresolvedNamedLambdaVariable` | Class | `(name_parts: Sequence[str])` |
| `sql.connect.expressions.UnresolvedRegex` | Class | `(col_name: str, plan_id: Optional[int])` |
| `sql.connect.expressions.UnresolvedStar` | Class | `(unparsed_target: Optional[str], plan_id: Optional[int])` |
| `sql.connect.expressions.WindowExpression` | Class | `(windowFunction: Expression, windowSpec: WindowSpec)` |
| `sql.connect.expressions.WindowSpec` | Class | `(partitionSpec: Sequence[Expression], orderSpec: Sequence[SortOrder], frame: Optional[WindowFrame])` |
| `sql.connect.expressions.WithField` | Class | `(structExpr: Expression, fieldName: str, valueExpr: Expression)` |
| `sql.connect.expressions.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.expressions.current_origin` | Function | `() -> threading.local` |
| `sql.connect.expressions.enum_to_value` | Function | `(value: Any) -> Any` |
| `sql.connect.expressions.is_timestamp_ntz_preferred` | Function | `() -> bool` |
| `sql.connect.expressions.proto` | Object | `` |
| `sql.connect.expressions.proto_schema_to_pyspark_data_type` | Function | `(schema: pb2.DataType) -> DataType` |
| `sql.connect.expressions.pyspark_types_to_proto_types` | Function | `(data_type: DataType) -> pb2.DataType` |
| `sql.connect.functions.AnalyzeArgument` | Class | `(dataType: DataType, value: Optional[Any], isTable: bool, isConstantExpression: bool)` |
| `sql.connect.functions.AnalyzeResult` | Class | `(schema: StructType, withSinglePartition: bool, partitionBy: Sequence[PartitioningColumn], orderBy: Sequence[OrderingColumn], select: Sequence[SelectedColumn])` |
| `sql.connect.functions.Any` | Object | `` |
| `sql.connect.functions.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.connect.functions.CallFunction` | Class | `(name: str, args: Sequence[Expression])` |
| `sql.connect.functions.Callable` | Object | `` |
| `sql.connect.functions.CaseWhen` | Class | `(branches: Sequence[Tuple[Expression, Expression]], else_value: Optional[Expression])` |
| `sql.connect.functions.Column` | Class | `(...)` |
| `sql.connect.functions.ColumnReference` | Class | `(unparsed_identifier: str, plan_id: Optional[int], is_metadata_column: bool)` |
| `sql.connect.functions.DataType` | Class | `(...)` |
| `sql.connect.functions.Expression` | Class | `()` |
| `sql.connect.functions.LambdaFunction` | Class | `(function: Expression, arguments: Sequence[UnresolvedNamedLambdaVariable])` |
| `sql.connect.functions.List` | Object | `` |
| `sql.connect.functions.LiteralExpression` | Class | `(value: Any, dataType: DataType)` |
| `sql.connect.functions.Mapping` | Object | `` |
| `sql.connect.functions.Optional` | Object | `` |
| `sql.connect.functions.PySparkTypeError` | Class | `(...)` |
| `sql.connect.functions.PySparkValueError` | Class | `(...)` |
| `sql.connect.functions.SQLExpression` | Class | `(expr: str)` |
| `sql.connect.functions.Sequence` | Object | `` |
| `sql.connect.functions.SortOrder` | Class | `(child: Expression, ascending: bool, nullsFirst: bool)` |
| `sql.connect.functions.StringType` | Class | `(collation: str)` |
| `sql.connect.functions.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.functions.TYPE_CHECKING` | Object | `` |
| `sql.connect.functions.Tuple` | Object | `` |
| `sql.connect.functions.Type` | Object | `` |
| `sql.connect.functions.Union` | Object | `` |
| `sql.connect.functions.UnresolvedFunction` | Class | `(name: str, args: Sequence[Expression], is_distinct: bool)` |
| `sql.connect.functions.UnresolvedNamedLambdaVariable` | Class | `(name_parts: Sequence[str])` |
| `sql.connect.functions.UnresolvedStar` | Class | `(unparsed_target: Optional[str], plan_id: Optional[int])` |
| `sql.connect.functions.ValuesView` | Object | `` |
| `sql.connect.functions.abs` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.acos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.acosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.add_months` | Function | `(start: ColumnOrName, months: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.aes_encrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], iv: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.aggregate` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.connect.functions.any_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.approxCountDistinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.connect.functions.approx_count_distinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.connect.functions.approx_percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.connect.functions.array` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.array_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.array_append` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.array_compact` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.array_contains` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.array_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.array_except` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.array_insert` | Function | `(arr: ColumnOrName, pos: Union[ColumnOrName, int], value: Any) -> Column` |
| `sql.connect.functions.array_intersect` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.array_join` | Function | `(col: ColumnOrName, delimiter: str, null_replacement: Optional[str]) -> Column` |
| `sql.connect.functions.array_max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.array_min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.array_position` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.array_prepend` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.array_remove` | Function | `(col: ColumnOrName, element: Any) -> Column` |
| `sql.connect.functions.array_repeat` | Function | `(col: ColumnOrName, count: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.array_size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.array_sort` | Function | `(col: ColumnOrName, comparator: Optional[Callable[[Column, Column], Column]]) -> Column` |
| `sql.connect.functions.array_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.arrays_overlap` | Function | `(a1: ColumnOrName, a2: ColumnOrName) -> Column` |
| `sql.connect.functions.arrays_zip` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.arrow_udf` | Function | `(f: ArrowScalarToScalarFunction \| DataTypeOrString \| ArrowScalarIterFunction \| (AtomicDataTypeOrString \| ArrayType), returnType: DataTypeOrString \| ArrowScalarUDFType \| (AtomicDataTypeOrString \| ArrayType) \| ArrowScalarIterUDFType, functionType: ArrowScalarUDFType \| ArrowScalarIterUDFType) -> UserDefinedFunctionLike \| Callable[[ArrowScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[ArrowScalarIterFunction], UserDefinedFunctionLike]` |
| `sql.connect.functions.arrow_udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.connect.functions.asc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.asc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.asc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.ascii` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.asin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.asinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.assert_true` | Function | `(col: ColumnOrName, errMsg: Optional[Union[Column, str]]) -> Column` |
| `sql.connect.functions.atan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.atan2` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.atanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.base64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bit_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bit_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bit_get` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.connect.functions.bit_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bit_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bit_xor` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitmap_and_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitmap_bit_position` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitmap_bucket_number` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitmap_construct_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitmap_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitmap_or_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitwiseNOT` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bitwise_not` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bool_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.bool_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.broadcast` | Function | `(df: DataFrame) -> DataFrame` |
| `sql.connect.functions.bround` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.btrim` | Function | `(str: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.bucket` | Function | `(numBuckets: Union[Column, int], col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.AnalyzeArgument` | Class | `(dataType: DataType, value: Optional[Any], isTable: bool, isConstantExpression: bool)` |
| `sql.connect.functions.builtin.AnalyzeResult` | Class | `(schema: StructType, withSinglePartition: bool, partitionBy: Sequence[PartitioningColumn], orderBy: Sequence[OrderingColumn], select: Sequence[SelectedColumn])` |
| `sql.connect.functions.builtin.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.connect.functions.builtin.CallFunction` | Class | `(name: str, args: Sequence[Expression])` |
| `sql.connect.functions.builtin.CaseWhen` | Class | `(branches: Sequence[Tuple[Expression, Expression]], else_value: Optional[Expression])` |
| `sql.connect.functions.builtin.Column` | Class | `(...)` |
| `sql.connect.functions.builtin.ColumnOrName` | Object | `` |
| `sql.connect.functions.builtin.ColumnReference` | Class | `(unparsed_identifier: str, plan_id: Optional[int], is_metadata_column: bool)` |
| `sql.connect.functions.builtin.DataFrame` | Class | `(...)` |
| `sql.connect.functions.builtin.DataType` | Class | `(...)` |
| `sql.connect.functions.builtin.DataTypeOrString` | Object | `` |
| `sql.connect.functions.builtin.Expression` | Class | `()` |
| `sql.connect.functions.builtin.LambdaFunction` | Class | `(function: Expression, arguments: Sequence[UnresolvedNamedLambdaVariable])` |
| `sql.connect.functions.builtin.LiteralExpression` | Class | `(value: Any, dataType: DataType)` |
| `sql.connect.functions.builtin.PySparkTypeError` | Class | `(...)` |
| `sql.connect.functions.builtin.PySparkValueError` | Class | `(...)` |
| `sql.connect.functions.builtin.SQLExpression` | Class | `(expr: str)` |
| `sql.connect.functions.builtin.SortOrder` | Class | `(child: Expression, ascending: bool, nullsFirst: bool)` |
| `sql.connect.functions.builtin.StringType` | Class | `(collation: str)` |
| `sql.connect.functions.builtin.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.functions.builtin.UnresolvedFunction` | Class | `(name: str, args: Sequence[Expression], is_distinct: bool)` |
| `sql.connect.functions.builtin.UnresolvedNamedLambdaVariable` | Class | `(name_parts: Sequence[str])` |
| `sql.connect.functions.builtin.UnresolvedStar` | Class | `(unparsed_target: Optional[str], plan_id: Optional[int])` |
| `sql.connect.functions.builtin.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.connect.functions.builtin.UserDefinedTableFunction` | Class | `(func: Type, returnType: Optional[Union[StructType, str]], name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.connect.functions.builtin.abs` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.acos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.acosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.add_months` | Function | `(start: ColumnOrName, months: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.aes_encrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], iv: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.aggregate` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.connect.functions.builtin.any_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.builtin.approxCountDistinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.connect.functions.builtin.approx_count_distinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.connect.functions.builtin.approx_percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.connect.functions.builtin.array` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.builtin.array_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_append` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.builtin.array_compact` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_contains` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.builtin.array_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_except` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_insert` | Function | `(arr: ColumnOrName, pos: Union[ColumnOrName, int], value: Any) -> Column` |
| `sql.connect.functions.builtin.array_intersect` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_join` | Function | `(col: ColumnOrName, delimiter: str, null_replacement: Optional[str]) -> Column` |
| `sql.connect.functions.builtin.array_max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_position` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.builtin.array_prepend` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.builtin.array_remove` | Function | `(col: ColumnOrName, element: Any) -> Column` |
| `sql.connect.functions.builtin.array_repeat` | Function | `(col: ColumnOrName, count: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.array_size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.array_sort` | Function | `(col: ColumnOrName, comparator: Optional[Callable[[Column, Column], Column]]) -> Column` |
| `sql.connect.functions.builtin.array_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.arrays_overlap` | Function | `(a1: ColumnOrName, a2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.arrays_zip` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.arrow_udf` | Function | `(f: ArrowScalarToScalarFunction \| DataTypeOrString \| ArrowScalarIterFunction \| (AtomicDataTypeOrString \| ArrayType), returnType: DataTypeOrString \| ArrowScalarUDFType \| (AtomicDataTypeOrString \| ArrayType) \| ArrowScalarIterUDFType, functionType: ArrowScalarUDFType \| ArrowScalarIterUDFType) -> UserDefinedFunctionLike \| Callable[[ArrowScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[ArrowScalarIterFunction], UserDefinedFunctionLike]` |
| `sql.connect.functions.builtin.arrow_udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.connect.functions.builtin.asc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.asc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.asc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.ascii` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.asin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.asinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.assert_true` | Function | `(col: ColumnOrName, errMsg: Optional[Union[Column, str]]) -> Column` |
| `sql.connect.functions.builtin.atan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.atan2` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.builtin.atanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.base64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bit_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bit_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bit_get` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bit_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bit_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bit_xor` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitmap_and_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitmap_bit_position` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitmap_bucket_number` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitmap_construct_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitmap_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitmap_or_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitwiseNOT` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bitwise_not` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bool_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.bool_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.broadcast` | Function | `(df: DataFrame) -> DataFrame` |
| `sql.connect.functions.builtin.bround` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.btrim` | Function | `(str: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.bucket` | Function | `(numBuckets: Union[Column, int], col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.call_function` | Function | `(funcName: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.call_udf` | Function | `(udfName: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.cardinality` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.cbrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.ceil` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.ceiling` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.char` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.char_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.character_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.functions.builtin.chr` | Function | `(n: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.coalesce` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.col` | Function | `(col: str) -> Column` |
| `sql.connect.functions.builtin.collate` | Function | `(col: ColumnOrName, collation: str) -> Column` |
| `sql.connect.functions.builtin.collation` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.collect_list` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.collect_set` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.column` | Object | `` |
| `sql.connect.functions.builtin.concat` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.concat_ws` | Function | `(sep: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.contains` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.conv` | Function | `(col: ColumnOrName, fromBase: int, toBase: int) -> Column` |
| `sql.connect.functions.builtin.convert_timezone` | Function | `(sourceTz: Optional[Column], targetTz: Column, sourceTs: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.corr` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.cos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.cosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.cot` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.countDistinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.count_distinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.count_if` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.count_min_sketch` | Function | `(col: ColumnOrName, eps: Union[Column, float], confidence: Union[Column, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.covar_pop` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.covar_samp` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.crc32` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.create_map` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.builtin.csc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.cume_dist` | Function | `() -> Column` |
| `sql.connect.functions.builtin.curdate` | Function | `() -> Column` |
| `sql.connect.functions.builtin.current_catalog` | Function | `() -> Column` |
| `sql.connect.functions.builtin.current_database` | Function | `() -> Column` |
| `sql.connect.functions.builtin.current_date` | Function | `() -> Column` |
| `sql.connect.functions.builtin.current_schema` | Function | `() -> Column` |
| `sql.connect.functions.builtin.current_time` | Function | `(precision: Optional[int]) -> Column` |
| `sql.connect.functions.builtin.current_timestamp` | Function | `() -> Column` |
| `sql.connect.functions.builtin.current_timezone` | Function | `() -> Column` |
| `sql.connect.functions.builtin.current_user` | Function | `() -> Column` |
| `sql.connect.functions.builtin.date_add` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.date_diff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.date_format` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.connect.functions.builtin.date_from_unix_date` | Function | `(days: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.date_part` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.date_sub` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.date_trunc` | Function | `(format: str, timestamp: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.dateadd` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.datediff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.datepart` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.day` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.dayname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.dayofmonth` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.dayofweek` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.dayofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.days` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.decode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.connect.functions.builtin.degrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.dense_rank` | Function | `() -> Column` |
| `sql.connect.functions.builtin.desc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.desc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.desc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.e` | Function | `() -> Column` |
| `sql.connect.functions.builtin.element_at` | Function | `(col: ColumnOrName, extraction: Any) -> Column` |
| `sql.connect.functions.builtin.elt` | Function | `(inputs: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.encode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.connect.functions.builtin.endswith` | Function | `(str: ColumnOrName, suffix: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.equal_null` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.every` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.exists` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.connect.functions.builtin.exp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.explode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.explode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.expm1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.expr` | Function | `(str: str) -> Column` |
| `sql.connect.functions.builtin.extract` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.factorial` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.filter` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.connect.functions.builtin.find_in_set` | Function | `(str: ColumnOrName, str_array: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.first` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.connect.functions.builtin.first_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.builtin.flatten` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.floor` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.forall` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.connect.functions.builtin.format_number` | Function | `(col: ColumnOrName, d: int) -> Column` |
| `sql.connect.functions.builtin.format_string` | Function | `(format: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.from_csv` | Function | `(col: ColumnOrName, schema: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.from_json` | Function | `(col: ColumnOrName, schema: Union[ArrayType, StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.from_unixtime` | Function | `(timestamp: ColumnOrName, format: str) -> Column` |
| `sql.connect.functions.builtin.from_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.connect.functions.builtin.from_xml` | Function | `(col: ColumnOrName, schema: Union[StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.get` | Function | `(col: ColumnOrName, index: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.get_json_object` | Function | `(col: ColumnOrName, path: str) -> Column` |
| `sql.connect.functions.builtin.getbit` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.greatest` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.grouping` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.grouping_id` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.hash` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.hex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.histogram_numeric` | Function | `(col: ColumnOrName, nBins: Column) -> Column` |
| `sql.connect.functions.builtin.hll_sketch_agg` | Function | `(col: ColumnOrName, lgConfigK: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.hll_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.hll_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, allowDifferentLgConfigK: Optional[bool]) -> Column` |
| `sql.connect.functions.builtin.hll_union_agg` | Function | `(col: ColumnOrName, allowDifferentLgConfigK: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.builtin.hour` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.hours` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.hypot` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.builtin.ifnull` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.ilike` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.connect.functions.builtin.initcap` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.inline` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.inline_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.input_file_block_length` | Function | `() -> Column` |
| `sql.connect.functions.builtin.input_file_block_start` | Function | `() -> Column` |
| `sql.connect.functions.builtin.input_file_name` | Function | `() -> Column` |
| `sql.connect.functions.builtin.instr` | Function | `(str: ColumnOrName, substr: Union[Column, str]) -> Column` |
| `sql.connect.functions.builtin.is_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.is_variant_null` | Function | `(v: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.isnan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.isnotnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.isnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.java_method` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.json_array_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.json_object_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.json_tuple` | Function | `(col: ColumnOrName, fields: str) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_agg_bigint` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_agg_double` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_agg_float` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_n_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_n_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_n_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_quantile_bigint` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_quantile_double` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_quantile_float` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_rank_bigint` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_rank_double` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_get_rank_float` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_merge_bigint` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_merge_double` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_merge_float` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_to_string_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_to_string_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kll_sketch_to_string_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.kurtosis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.lag` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.connect.functions.builtin.last` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.connect.functions.builtin.last_day` | Function | `(date: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.last_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.builtin.lcase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.lead` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.connect.functions.builtin.least` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.left` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.levenshtein` | Function | `(left: ColumnOrName, right: ColumnOrName, threshold: Optional[int]) -> Column` |
| `sql.connect.functions.builtin.like` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.connect.functions.builtin.listagg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.builtin.listagg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.builtin.lit` | Function | `(col: Any) -> Column` |
| `sql.connect.functions.builtin.ln` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.localtimestamp` | Function | `() -> Column` |
| `sql.connect.functions.builtin.locate` | Function | `(substr: str, str: ColumnOrName, pos: int) -> Column` |
| `sql.connect.functions.builtin.log` | Function | `(arg1: Union[ColumnOrName, float], arg2: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.log10` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.log1p` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.log2` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.lower` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.lpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.connect.functions.builtin.ltrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.make_date` | Function | `(year: ColumnOrName, month: ColumnOrName, day: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.make_dt_interval` | Function | `(days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.make_time` | Function | `(hour: ColumnOrName, minute: ColumnOrName, second: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.make_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.make_ym_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.map_concat` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.builtin.map_contains_key` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.builtin.map_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.map_filter` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.functions.builtin.map_from_arrays` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.map_from_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.map_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.map_values` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.map_zip_with` | Function | `(col1: ColumnOrName, col2: ColumnOrName, f: Callable[[Column, Column, Column], Column]) -> Column` |
| `sql.connect.functions.builtin.mask` | Function | `(col: ColumnOrName, upperChar: Optional[ColumnOrName], lowerChar: Optional[ColumnOrName], digitChar: Optional[ColumnOrName], otherChar: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.max_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.md5` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.mean` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.median` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.min_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.minute` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.mode` | Function | `(col: ColumnOrName, deterministic: bool) -> Column` |
| `sql.connect.functions.builtin.monotonically_increasing_id` | Function | `() -> Column` |
| `sql.connect.functions.builtin.month` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.monthname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.months` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.months_between` | Function | `(date1: ColumnOrName, date2: ColumnOrName, roundOff: bool) -> Column` |
| `sql.connect.functions.builtin.named_struct` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.nanvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.negate` | Object | `` |
| `sql.connect.functions.builtin.negative` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.next_day` | Function | `(date: ColumnOrName, dayOfWeek: str) -> Column` |
| `sql.connect.functions.builtin.now` | Function | `() -> Column` |
| `sql.connect.functions.builtin.nth_value` | Function | `(col: ColumnOrName, offset: int, ignoreNulls: Optional[bool]) -> Column` |
| `sql.connect.functions.builtin.ntile` | Function | `(n: int) -> Column` |
| `sql.connect.functions.builtin.nullif` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.nullifzero` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.nvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.nvl2` | Function | `(col1: ColumnOrName, col2: ColumnOrName, col3: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.octet_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.overlay` | Function | `(src: ColumnOrName, replace: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `sql.connect.functions.builtin.parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.percent_rank` | Function | `() -> Column` |
| `sql.connect.functions.builtin.percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], frequency: Union[Column, int]) -> Column` |
| `sql.connect.functions.builtin.percentile_approx` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.connect.functions.builtin.pi` | Function | `() -> Column` |
| `sql.connect.functions.builtin.pmod` | Function | `(dividend: Union[ColumnOrName, float], divisor: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.builtin.posexplode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.posexplode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.position` | Function | `(substr: ColumnOrName, str: ColumnOrName, start: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.positive` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.pow` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.builtin.power` | Object | `` |
| `sql.connect.functions.builtin.printf` | Function | `(format: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.product` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.pysparkfuncs` | Object | `` |
| `sql.connect.functions.builtin.quarter` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.quote` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.radians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.raise_error` | Function | `(errMsg: Union[Column, str]) -> Column` |
| `sql.connect.functions.builtin.rand` | Function | `(seed: Optional[int]) -> Column` |
| `sql.connect.functions.builtin.randn` | Function | `(seed: Optional[int]) -> Column` |
| `sql.connect.functions.builtin.random` | Object | `` |
| `sql.connect.functions.builtin.randstr` | Function | `(length: Union[Column, int], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.rank` | Function | `() -> Column` |
| `sql.connect.functions.builtin.reduce` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.connect.functions.builtin.reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regexp` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regexp_count` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regexp_extract` | Function | `(str: ColumnOrName, pattern: str, idx: int) -> Column` |
| `sql.connect.functions.builtin.regexp_extract_all` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.regexp_instr` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.regexp_like` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regexp_replace` | Function | `(string: ColumnOrName, pattern: Union[str, Column], replacement: Union[str, Column]) -> Column` |
| `sql.connect.functions.builtin.regexp_substr` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_avgx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_avgy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_count` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_intercept` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_r2` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_slope` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_sxx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_sxy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.regr_syy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.repeat` | Function | `(col: ColumnOrName, n: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.replace` | Function | `(src: ColumnOrName, search: ColumnOrName, replace: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.reverse` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.right` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.rint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.rlike` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.round` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.row_number` | Function | `() -> Column` |
| `sql.connect.functions.builtin.rpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.connect.functions.builtin.rtrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.schema_of_csv` | Function | `(csv: Union[str, Column], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.schema_of_json` | Function | `(json: Union[str, Column], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.schema_of_variant` | Function | `(v: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.schema_of_variant_agg` | Function | `(v: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.schema_of_xml` | Function | `(xml: Union[str, Column], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.sec` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.second` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sentences` | Function | `(string: ColumnOrName, language: Optional[ColumnOrName], country: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.sequence` | Function | `(start: ColumnOrName, stop: ColumnOrName, step: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.session_user` | Function | `() -> Column` |
| `sql.connect.functions.builtin.session_window` | Function | `(timeColumn: ColumnOrName, gapDuration: Union[Column, str]) -> Column` |
| `sql.connect.functions.builtin.sha` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sha1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sha2` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.builtin.shiftLeft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.builtin.shiftRight` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.builtin.shiftRightUnsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.builtin.shiftleft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.builtin.shiftright` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.builtin.shiftrightunsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.builtin.shuffle` | Function | `(col: ColumnOrName, seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.sign` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.signum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.skewness` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.slice` | Function | `(x: ColumnOrName, start: Union[ColumnOrName, int], length: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.some` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sort_array` | Function | `(col: ColumnOrName, asc: bool) -> Column` |
| `sql.connect.functions.builtin.soundex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.spark_partition_id` | Function | `() -> Column` |
| `sql.connect.functions.builtin.split` | Function | `(str: ColumnOrName, pattern: Union[Column, str], limit: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.split_part` | Function | `(src: ColumnOrName, delimiter: ColumnOrName, partNum: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sqrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.st_asbinary` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.st_geogfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.st_geomfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.st_setsrid` | Function | `(geo: ColumnOrName, srid: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.st_srid` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.stack` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.startswith` | Function | `(str: ColumnOrName, prefix: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.std` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.stddev` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.stddev_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.stddev_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.str_to_map` | Function | `(text: ColumnOrName, pairDelim: Optional[ColumnOrName], keyValueDelim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.string_agg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.builtin.string_agg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.builtin.struct` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.builtin.substr` | Function | `(str: ColumnOrName, pos: ColumnOrName, len: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.substring` | Function | `(str: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.substring_index` | Function | `(str: ColumnOrName, delim: str, count: int) -> Column` |
| `sql.connect.functions.builtin.sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sumDistinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.sum_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.tan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.tanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.theta_difference` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.theta_intersection` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.theta_intersection_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.theta_sketch_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.theta_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.theta_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.theta_union_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.builtin.time_diff` | Function | `(unit: ColumnOrName, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.time_trunc` | Function | `(unit: ColumnOrName, time: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.timestamp_add` | Function | `(unit: str, quantity: ColumnOrName, ts: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.timestamp_diff` | Function | `(unit: str, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.timestamp_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.timestamp_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.timestamp_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.toDegrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.toRadians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.to_char` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.to_csv` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.connect.functions.builtin.to_json` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.to_timestamp` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.connect.functions.builtin.to_timestamp_ltz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.to_timestamp_ntz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.to_unix_timestamp` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.to_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.connect.functions.builtin.to_varchar` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.to_variant_object` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.to_xml` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.builtin.transform` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.connect.functions.builtin.transform_keys` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.functions.builtin.transform_values` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.functions.builtin.translate` | Function | `(srcCol: ColumnOrName, matching: str, replace: str) -> Column` |
| `sql.connect.functions.builtin.trim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.trunc` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.connect.functions.builtin.try_add` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_divide` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_element_at` | Function | `(col: ColumnOrName, extraction: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_mod` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_multiply` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_subtract` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.connect.functions.builtin.try_to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_to_timestamp` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.builtin.try_url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.try_variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.connect.functions.builtin.typeof` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.ucase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.udf` | Function | `(f: Optional[Union[Callable[..., Any], DataTypeOrString]], returnType: DataTypeOrString, useArrow: Optional[bool]) -> Union[UserDefinedFunctionLike, Callable[[Callable[..., Any]], UserDefinedFunctionLike]]` |
| `sql.connect.functions.builtin.udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]], useArrow: Optional[bool]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.connect.functions.builtin.unbase64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.unhex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.uniform` | Function | `(min: Union[Column, int, float], max: Union[Column, int, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.unix_date` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.unix_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.unix_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.unix_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.unix_timestamp` | Function | `(timestamp: Optional[ColumnOrName], format: str) -> Column` |
| `sql.connect.functions.builtin.unwrap_udt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.upper` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.url_encode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.user` | Function | `() -> Column` |
| `sql.connect.functions.builtin.uuid` | Function | `(seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.builtin.validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.var_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.var_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.variance` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.connect.functions.builtin.version` | Function | `() -> Column` |
| `sql.connect.functions.builtin.weekday` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.weekofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.when` | Function | `(condition: Column, value: Any) -> Column` |
| `sql.connect.functions.builtin.width_bucket` | Function | `(v: ColumnOrName, min: ColumnOrName, max: ColumnOrName, numBucket: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.builtin.window` | Function | `(timeColumn: ColumnOrName, windowDuration: str, slideDuration: Optional[str], startTime: Optional[str]) -> Column` |
| `sql.connect.functions.builtin.window_time` | Function | `(windowColumn: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_boolean` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_double` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_float` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_int` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_long` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_number` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_short` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xpath_string` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.xxhash64` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.year` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.years` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.zeroifnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.builtin.zip_with` | Function | `(left: ColumnOrName, right: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.functions.call_function` | Function | `(funcName: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.call_udf` | Function | `(udfName: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.cardinality` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.cast` | Object | `` |
| `sql.connect.functions.cbrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.ceil` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.ceiling` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.char` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.char_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.character_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.functions.chr` | Function | `(n: ColumnOrName) -> Column` |
| `sql.connect.functions.coalesce` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.col` | Function | `(col: str) -> Column` |
| `sql.connect.functions.collate` | Function | `(col: ColumnOrName, collation: str) -> Column` |
| `sql.connect.functions.collation` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.collect_list` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.collect_set` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.column` | Object | `` |
| `sql.connect.functions.concat` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.concat_ws` | Function | `(sep: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.contains` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.conv` | Function | `(col: ColumnOrName, fromBase: int, toBase: int) -> Column` |
| `sql.connect.functions.convert_timezone` | Function | `(sourceTz: Optional[Column], targetTz: Column, sourceTs: ColumnOrName) -> Column` |
| `sql.connect.functions.corr` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.cos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.cosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.cot` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.countDistinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.count_distinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.count_if` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.count_min_sketch` | Function | `(col: ColumnOrName, eps: Union[Column, float], confidence: Union[Column, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.covar_pop` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.covar_samp` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.crc32` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.create_map` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.csc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.cume_dist` | Function | `() -> Column` |
| `sql.connect.functions.curdate` | Function | `() -> Column` |
| `sql.connect.functions.current_catalog` | Function | `() -> Column` |
| `sql.connect.functions.current_database` | Function | `() -> Column` |
| `sql.connect.functions.current_date` | Function | `() -> Column` |
| `sql.connect.functions.current_schema` | Function | `() -> Column` |
| `sql.connect.functions.current_time` | Function | `(precision: Optional[int]) -> Column` |
| `sql.connect.functions.current_timestamp` | Function | `() -> Column` |
| `sql.connect.functions.current_timezone` | Function | `() -> Column` |
| `sql.connect.functions.current_user` | Function | `() -> Column` |
| `sql.connect.functions.date_add` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.date_diff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.connect.functions.date_format` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.connect.functions.date_from_unix_date` | Function | `(days: ColumnOrName) -> Column` |
| `sql.connect.functions.date_part` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.connect.functions.date_sub` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.date_trunc` | Function | `(format: str, timestamp: ColumnOrName) -> Column` |
| `sql.connect.functions.dateadd` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.datediff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.connect.functions.datepart` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.connect.functions.day` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.dayname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.dayofmonth` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.dayofweek` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.dayofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.days` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.decimal` | Object | `` |
| `sql.connect.functions.decode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.connect.functions.degrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.dense_rank` | Function | `() -> Column` |
| `sql.connect.functions.desc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.desc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.desc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.e` | Function | `() -> Column` |
| `sql.connect.functions.element_at` | Function | `(col: ColumnOrName, extraction: Any) -> Column` |
| `sql.connect.functions.elt` | Function | `(inputs: ColumnOrName) -> Column` |
| `sql.connect.functions.encode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.connect.functions.endswith` | Function | `(str: ColumnOrName, suffix: ColumnOrName) -> Column` |
| `sql.connect.functions.equal_null` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.every` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.exists` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.connect.functions.exp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.explode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.explode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.expm1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.expr` | Function | `(str: str) -> Column` |
| `sql.connect.functions.extract` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.connect.functions.factorial` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.filter` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.connect.functions.find_in_set` | Function | `(str: ColumnOrName, str_array: ColumnOrName) -> Column` |
| `sql.connect.functions.first` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.connect.functions.first_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.flatten` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.floor` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.forall` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.connect.functions.format_number` | Function | `(col: ColumnOrName, d: int) -> Column` |
| `sql.connect.functions.format_string` | Function | `(format: str, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.from_csv` | Function | `(col: ColumnOrName, schema: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.from_json` | Function | `(col: ColumnOrName, schema: Union[ArrayType, StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.from_unixtime` | Function | `(timestamp: ColumnOrName, format: str) -> Column` |
| `sql.connect.functions.from_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.connect.functions.from_xml` | Function | `(col: ColumnOrName, schema: Union[StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.functools` | Object | `` |
| `sql.connect.functions.get` | Function | `(col: ColumnOrName, index: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.get_json_object` | Function | `(col: ColumnOrName, path: str) -> Column` |
| `sql.connect.functions.getbit` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.connect.functions.greatest` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.grouping` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.grouping_id` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.hash` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.hex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.histogram_numeric` | Function | `(col: ColumnOrName, nBins: Column) -> Column` |
| `sql.connect.functions.hll_sketch_agg` | Function | `(col: ColumnOrName, lgConfigK: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.hll_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.hll_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, allowDifferentLgConfigK: Optional[bool]) -> Column` |
| `sql.connect.functions.hll_union_agg` | Function | `(col: ColumnOrName, allowDifferentLgConfigK: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.hour` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.hours` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.hypot` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.ifnull` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.ilike` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.connect.functions.initcap` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.inline` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.inline_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.input_file_block_length` | Function | `() -> Column` |
| `sql.connect.functions.input_file_block_start` | Function | `() -> Column` |
| `sql.connect.functions.input_file_name` | Function | `() -> Column` |
| `sql.connect.functions.inspect` | Object | `` |
| `sql.connect.functions.instr` | Function | `(str: ColumnOrName, substr: Union[Column, str]) -> Column` |
| `sql.connect.functions.is_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.is_variant_null` | Function | `(v: ColumnOrName) -> Column` |
| `sql.connect.functions.isnan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.isnotnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.isnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.java_method` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.json_array_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.json_object_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.json_tuple` | Function | `(col: ColumnOrName, fields: str) -> Column` |
| `sql.connect.functions.kll_sketch_agg_bigint` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.kll_sketch_agg_double` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.kll_sketch_agg_float` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.kll_sketch_get_n_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_n_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_n_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_quantile_bigint` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_quantile_double` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_quantile_float` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_rank_bigint` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_rank_double` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_get_rank_float` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_merge_bigint` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_merge_double` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_merge_float` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_to_string_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_to_string_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.kll_sketch_to_string_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.kurtosis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.lag` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.connect.functions.last` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.connect.functions.last_day` | Function | `(date: ColumnOrName) -> Column` |
| `sql.connect.functions.last_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.connect.functions.lcase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.lead` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.connect.functions.least` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.left` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.connect.functions.length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.levenshtein` | Function | `(left: ColumnOrName, right: ColumnOrName, threshold: Optional[int]) -> Column` |
| `sql.connect.functions.like` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.connect.functions.listagg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.listagg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.lit` | Function | `(col: Any) -> Column` |
| `sql.connect.functions.ln` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.localtimestamp` | Function | `() -> Column` |
| `sql.connect.functions.locate` | Function | `(substr: str, str: ColumnOrName, pos: int) -> Column` |
| `sql.connect.functions.log` | Function | `(arg1: Union[ColumnOrName, float], arg2: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.log10` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.log1p` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.log2` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.lower` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.lpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.connect.functions.ltrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.make_date` | Function | `(year: ColumnOrName, month: ColumnOrName, day: ColumnOrName) -> Column` |
| `sql.connect.functions.make_dt_interval` | Function | `(days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.make_time` | Function | `(hour: ColumnOrName, minute: ColumnOrName, second: ColumnOrName) -> Column` |
| `sql.connect.functions.make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.make_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.make_ym_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.map_concat` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.map_contains_key` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.connect.functions.map_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.map_filter` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.functions.map_from_arrays` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.map_from_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.map_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.map_values` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.map_zip_with` | Function | `(col1: ColumnOrName, col2: ColumnOrName, f: Callable[[Column, Column, Column], Column]) -> Column` |
| `sql.connect.functions.mask` | Function | `(col: ColumnOrName, upperChar: Optional[ColumnOrName], lowerChar: Optional[ColumnOrName], digitChar: Optional[ColumnOrName], otherChar: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.max_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.connect.functions.md5` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.mean` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.median` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.min_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.connect.functions.minute` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.mode` | Function | `(col: ColumnOrName, deterministic: bool) -> Column` |
| `sql.connect.functions.monotonically_increasing_id` | Function | `() -> Column` |
| `sql.connect.functions.month` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.monthname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.months` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.months_between` | Function | `(date1: ColumnOrName, date2: ColumnOrName, roundOff: bool) -> Column` |
| `sql.connect.functions.named_struct` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.nanvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.negate` | Object | `` |
| `sql.connect.functions.negative` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.next_day` | Function | `(date: ColumnOrName, dayOfWeek: str) -> Column` |
| `sql.connect.functions.now` | Function | `() -> Column` |
| `sql.connect.functions.np` | Object | `` |
| `sql.connect.functions.nth_value` | Function | `(col: ColumnOrName, offset: int, ignoreNulls: Optional[bool]) -> Column` |
| `sql.connect.functions.ntile` | Function | `(n: int) -> Column` |
| `sql.connect.functions.nullif` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.nullifzero` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.nvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.nvl2` | Function | `(col1: ColumnOrName, col2: ColumnOrName, col3: ColumnOrName) -> Column` |
| `sql.connect.functions.octet_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.overlay` | Function | `(src: ColumnOrName, replace: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.overload` | Object | `` |
| `sql.connect.functions.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `sql.connect.functions.parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.partitioning.Column` | Class | `(...)` |
| `sql.connect.functions.partitioning.ColumnOrName` | Object | `` |
| `sql.connect.functions.partitioning.PySparkTypeError` | Class | `(...)` |
| `sql.connect.functions.partitioning.bucket` | Function | `(numBuckets: Union[Column, int], col: ColumnOrName) -> Column` |
| `sql.connect.functions.partitioning.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.functions.partitioning.days` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.partitioning.hours` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.partitioning.lit` | Function | `(col: Any) -> Column` |
| `sql.connect.functions.partitioning.months` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.partitioning.pysparkfuncs` | Object | `` |
| `sql.connect.functions.partitioning.years` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.percent_rank` | Function | `() -> Column` |
| `sql.connect.functions.percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], frequency: Union[Column, int]) -> Column` |
| `sql.connect.functions.percentile_approx` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.connect.functions.pi` | Function | `() -> Column` |
| `sql.connect.functions.pmod` | Function | `(dividend: Union[ColumnOrName, float], divisor: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.posexplode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.posexplode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.position` | Function | `(substr: ColumnOrName, str: ColumnOrName, start: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.positive` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.pow` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.connect.functions.power` | Object | `` |
| `sql.connect.functions.printf` | Function | `(format: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.connect.functions.product` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.py_random` | Object | `` |
| `sql.connect.functions.pysparkfuncs` | Object | `` |
| `sql.connect.functions.quarter` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.quote` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.radians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.raise_error` | Function | `(errMsg: Union[Column, str]) -> Column` |
| `sql.connect.functions.rand` | Function | `(seed: Optional[int]) -> Column` |
| `sql.connect.functions.randn` | Function | `(seed: Optional[int]) -> Column` |
| `sql.connect.functions.random` | Object | `` |
| `sql.connect.functions.randstr` | Function | `(length: Union[Column, int], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.rank` | Function | `() -> Column` |
| `sql.connect.functions.reduce` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.connect.functions.reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.regexp` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.regexp_count` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.regexp_extract` | Function | `(str: ColumnOrName, pattern: str, idx: int) -> Column` |
| `sql.connect.functions.regexp_extract_all` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.regexp_instr` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.regexp_like` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.regexp_replace` | Function | `(string: ColumnOrName, pattern: Union[str, Column], replacement: Union[str, Column]) -> Column` |
| `sql.connect.functions.regexp_substr` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_avgx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_avgy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_count` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_intercept` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_r2` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_slope` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_sxx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_sxy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.regr_syy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.connect.functions.repeat` | Function | `(col: ColumnOrName, n: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.replace` | Function | `(src: ColumnOrName, search: ColumnOrName, replace: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.reverse` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.right` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.connect.functions.rint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.rlike` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.connect.functions.round` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.row_number` | Function | `() -> Column` |
| `sql.connect.functions.rpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.connect.functions.rtrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.schema_of_csv` | Function | `(csv: Union[str, Column], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.schema_of_json` | Function | `(json: Union[str, Column], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.schema_of_variant` | Function | `(v: ColumnOrName) -> Column` |
| `sql.connect.functions.schema_of_variant_agg` | Function | `(v: ColumnOrName) -> Column` |
| `sql.connect.functions.schema_of_xml` | Function | `(xml: Union[str, Column], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.sec` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.second` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sentences` | Function | `(string: ColumnOrName, language: Optional[ColumnOrName], country: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.sequence` | Function | `(start: ColumnOrName, stop: ColumnOrName, step: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.session_user` | Function | `() -> Column` |
| `sql.connect.functions.session_window` | Function | `(timeColumn: ColumnOrName, gapDuration: Union[Column, str]) -> Column` |
| `sql.connect.functions.sha` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sha1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sha2` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.shiftLeft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.shiftRight` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.shiftRightUnsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.shiftleft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.shiftright` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.shiftrightunsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.connect.functions.should_test_connect` | Object | `` |
| `sql.connect.functions.shuffle` | Function | `(col: ColumnOrName, seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.sign` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.signum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.skewness` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.slice` | Function | `(x: ColumnOrName, start: Union[ColumnOrName, int], length: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.some` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sort_array` | Function | `(col: ColumnOrName, asc: bool) -> Column` |
| `sql.connect.functions.soundex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.spark_partition_id` | Function | `() -> Column` |
| `sql.connect.functions.split` | Function | `(str: ColumnOrName, pattern: Union[Column, str], limit: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.split_part` | Function | `(src: ColumnOrName, delimiter: ColumnOrName, partNum: ColumnOrName) -> Column` |
| `sql.connect.functions.sqrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.st_asbinary` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.connect.functions.st_geogfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.connect.functions.st_geomfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.connect.functions.st_setsrid` | Function | `(geo: ColumnOrName, srid: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.st_srid` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.connect.functions.stack` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.startswith` | Function | `(str: ColumnOrName, prefix: ColumnOrName) -> Column` |
| `sql.connect.functions.std` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.stddev` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.stddev_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.stddev_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.str_to_map` | Function | `(text: ColumnOrName, pairDelim: Optional[ColumnOrName], keyValueDelim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.string_agg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.string_agg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.connect.functions.struct` | Function | `(cols: Union[ColumnOrName, Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column` |
| `sql.connect.functions.substr` | Function | `(str: ColumnOrName, pos: ColumnOrName, len: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.substring` | Function | `(str: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.substring_index` | Function | `(str: ColumnOrName, delim: str, count: int) -> Column` |
| `sql.connect.functions.sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sumDistinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sum_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.sys` | Object | `` |
| `sql.connect.functions.tan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.tanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.theta_difference` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.theta_intersection` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.connect.functions.theta_intersection_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.theta_sketch_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.theta_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.theta_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.theta_union_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.connect.functions.time_diff` | Function | `(unit: ColumnOrName, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.connect.functions.time_trunc` | Function | `(unit: ColumnOrName, time: ColumnOrName) -> Column` |
| `sql.connect.functions.timestamp_add` | Function | `(unit: str, quantity: ColumnOrName, ts: ColumnOrName) -> Column` |
| `sql.connect.functions.timestamp_diff` | Function | `(unit: str, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.connect.functions.timestamp_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.timestamp_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.timestamp_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.toDegrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.toRadians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.to_char` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.to_csv` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.connect.functions.to_json` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.to_timestamp` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.connect.functions.to_timestamp_ltz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.to_timestamp_ntz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.to_unix_timestamp` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.to_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.connect.functions.to_varchar` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.to_variant_object` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.to_xml` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.connect.functions.transform` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.connect.functions.transform_keys` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.functions.transform_values` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.functions.translate` | Function | `(srcCol: ColumnOrName, matching: str, replace: str) -> Column` |
| `sql.connect.functions.trim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.trunc` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.connect.functions.try_add` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.try_aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.try_divide` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.try_element_at` | Function | `(col: ColumnOrName, extraction: ColumnOrName) -> Column` |
| `sql.connect.functions.try_make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_mod` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.try_multiply` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.try_parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.try_parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.try_subtract` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.connect.functions.try_sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.try_to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.connect.functions.try_to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.connect.functions.try_to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_to_timestamp` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.connect.functions.try_url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.try_validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.try_variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.connect.functions.typeof` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.ucase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.udf` | Function | `(f: Optional[Union[Callable[..., Any], DataTypeOrString]], returnType: DataTypeOrString, useArrow: Optional[bool]) -> Union[UserDefinedFunctionLike, Callable[[Callable[..., Any]], UserDefinedFunctionLike]]` |
| `sql.connect.functions.udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]], useArrow: Optional[bool]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.connect.functions.unbase64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.unhex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.uniform` | Function | `(min: Union[Column, int, float], max: Union[Column, int, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.unix_date` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.unix_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.unix_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.unix_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.unix_timestamp` | Function | `(timestamp: Optional[ColumnOrName], format: str) -> Column` |
| `sql.connect.functions.unwrap_udt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.upper` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.url_encode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.user` | Function | `() -> Column` |
| `sql.connect.functions.uuid` | Function | `(seed: Optional[Union[Column, int]]) -> Column` |
| `sql.connect.functions.validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.connect.functions.var_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.var_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.variance` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.connect.functions.version` | Function | `() -> Column` |
| `sql.connect.functions.warnings` | Object | `` |
| `sql.connect.functions.weekday` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.weekofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.when` | Function | `(condition: Column, value: Any) -> Column` |
| `sql.connect.functions.width_bucket` | Function | `(v: ColumnOrName, min: ColumnOrName, max: ColumnOrName, numBucket: Union[ColumnOrName, int]) -> Column` |
| `sql.connect.functions.window` | Function | `(timeColumn: ColumnOrName, windowDuration: str, slideDuration: Optional[str], startTime: Optional[str]) -> Column` |
| `sql.connect.functions.window_time` | Function | `(windowColumn: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_boolean` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_double` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_float` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_int` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_long` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_number` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_short` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xpath_string` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.connect.functions.xxhash64` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.connect.functions.year` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.years` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.zeroifnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.connect.functions.zip_with` | Function | `(left: ColumnOrName, right: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.connect.group.ArrowCogroupedMapFunction` | Object | `` |
| `sql.connect.group.ArrowGroupedMapFunction` | Object | `` |
| `sql.connect.group.Column` | Class | `(...)` |
| `sql.connect.group.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.group.F` | Object | `` |
| `sql.connect.group.GroupedData` | Class | `(df: DataFrame, group_type: str, grouping_cols: Sequence[Column], pivot_col: Optional[Column], pivot_values: Optional[Sequence[LiteralType]], grouping_sets: Optional[Sequence[Sequence[Column]]])` |
| `sql.connect.group.GroupedMapPandasUserDefinedFunction` | Object | `` |
| `sql.connect.group.LiteralType` | Object | `` |
| `sql.connect.group.NumericType` | Class | `(...)` |
| `sql.connect.group.PandasCogroupedMapFunction` | Object | `` |
| `sql.connect.group.PandasCogroupedOps` | Class | `(gd1: GroupedData, gd2: GroupedData)` |
| `sql.connect.group.PandasGroupedMapFunction` | Object | `` |
| `sql.connect.group.PandasGroupedMapFunctionWithState` | Object | `` |
| `sql.connect.group.PySparkGroupedData` | Class | `(jgd: JavaObject, df: DataFrame)` |
| `sql.connect.group.PySparkNotImplementedError` | Class | `(...)` |
| `sql.connect.group.PySparkPandasCogroupedOps` | Class | `(gd1: GroupedData, gd2: GroupedData)` |
| `sql.connect.group.PySparkTypeError` | Class | `(...)` |
| `sql.connect.group.PythonEvalType` | Class | `(...)` |
| `sql.connect.group.StatefulProcessor` | Class | `(...)` |
| `sql.connect.group.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.group.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.group.infer_group_arrow_eval_type_from_func` | Function | `(f: ArrowGroupedMapFunction) -> Optional[Union[ArrowGroupedMapUDFType, ArrowGroupedMapIterUDFType]]` |
| `sql.connect.group.plan` | Object | `` |
| `sql.connect.logging.PySparkLogger` | Class | `(name: str)` |
| `sql.connect.logging.configureLogging` | Function | `(level: Optional[str]) -> logging.Logger` |
| `sql.connect.logging.getLogLevel` | Function | `() -> Optional[int]` |
| `sql.connect.logging.logger` | Object | `` |
| `sql.connect.merge.Column` | Class | `(expr: Expression)` |
| `sql.connect.merge.ExecutionInfo` | Class | `(metrics: Optional[list[PlanMetrics]], obs: Optional[Sequence[ObservedMetrics]])` |
| `sql.connect.merge.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `sql.connect.merge.MergeIntoWriter` | Class | `(plan: LogicalPlan, session: SparkSession, table: str, condition: Column, callback: Optional[Callable[[ExecutionInfo], None]])` |
| `sql.connect.merge.PySparkMergeIntoWriter` | Class | `(df: DataFrame, table: str, condition: Column)` |
| `sql.connect.merge.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.merge.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.merge.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.merge.expr` | Function | `(str: str) -> Column` |
| `sql.connect.merge.proto` | Object | `` |
| `sql.connect.observation.Column` | Class | `(...)` |
| `sql.connect.observation.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.observation.IllegalArgumentException` | Class | `(...)` |
| `sql.connect.observation.Observation` | Class | `(name: Optional[str])` |
| `sql.connect.observation.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.connect.observation.PySparkObservation` | Class | `(name: Optional[str])` |
| `sql.connect.observation.PySparkTypeError` | Class | `(...)` |
| `sql.connect.observation.PySparkValueError` | Class | `(...)` |
| `sql.connect.observation.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.observation.plan` | Object | `` |
| `sql.connect.plan.Aggregate` | Class | `(child: Optional[LogicalPlan], group_type: str, grouping_cols: Sequence[Column], aggregate_cols: Sequence[Column], pivot_col: Optional[Column], pivot_values: Optional[Sequence[Column]], grouping_sets: Optional[Sequence[Sequence[Column]]])` |
| `sql.connect.plan.AnalysisException` | Class | `(...)` |
| `sql.connect.plan.ApplyInPandasWithState` | Class | `(child: Optional[LogicalPlan], grouping_cols: Sequence[Column], function: UserDefinedFunction, output_schema: str, state_schema: str, output_mode: str, timeout_conf: str, cols: List[str])` |
| `sql.connect.plan.AsOfJoin` | Class | `(left: LogicalPlan, right: LogicalPlan, left_as_of: Column, right_as_of: Column, on: Optional[Union[str, List[str], Column, List[Column]]], how: str, tolerance: Optional[Column], allow_exact_matches: bool, direction: str)` |
| `sql.connect.plan.BaseTransformWithStateInPySpark` | Class | `(child: Optional[LogicalPlan], grouping_cols: Sequence[Column], function: UserDefinedFunction, output_schema: Union[DataType, str], output_mode: str, time_mode: str, event_time_col_name: str, cols: List[str], initial_state_plan: Optional[LogicalPlan], initial_state_grouping_cols: Optional[Sequence[Column]])` |
| `sql.connect.plan.CacheTable` | Class | `(table_name: str, storage_level: Optional[StorageLevel])` |
| `sql.connect.plan.CachedRelation` | Class | `(plan: proto.Relation)` |
| `sql.connect.plan.CachedRemoteRelation` | Class | `(relation_id: str, spark_session: SparkSession)` |
| `sql.connect.plan.Checkpoint` | Class | `(child: Optional[LogicalPlan], local: bool, eager: bool, storage_level: Optional[StorageLevel])` |
| `sql.connect.plan.ChunkedCachedLocalRelation` | Class | `(data_hashes: list[str], schema_hash: Optional[str])` |
| `sql.connect.plan.ClearCache` | Class | `()` |
| `sql.connect.plan.CloudPickleSerializer` | Class | `(...)` |
| `sql.connect.plan.CoGroupMap` | Class | `(input: Optional[LogicalPlan], input_grouping_cols: Sequence[Column], other: Optional[LogicalPlan], other_grouping_cols: Sequence[Column], function: UserDefinedFunction)` |
| `sql.connect.plan.CollectMetrics` | Class | `(child: Optional[LogicalPlan], observation: Union[str, Observation], exprs: List[Column])` |
| `sql.connect.plan.Column` | Class | `(...)` |
| `sql.connect.plan.CommonInlineUserDefinedDataSource` | Class | `(name: str, data_source: PythonDataSource)` |
| `sql.connect.plan.CommonInlineUserDefinedTableFunction` | Class | `(function_name: str, function: PythonUDTF, deterministic: bool, arguments: Sequence[Expression])` |
| `sql.connect.plan.CreateTable` | Class | `(table_name: str, path: str, source: Optional[str], description: Optional[str], schema: Optional[DataType], options: Mapping[str, str])` |
| `sql.connect.plan.CreateView` | Class | `(child: Optional[LogicalPlan], name: str, is_global: bool, replace: bool)` |
| `sql.connect.plan.CurrentCatalog` | Class | `()` |
| `sql.connect.plan.CurrentDatabase` | Class | `()` |
| `sql.connect.plan.DataSource` | Class | `(format: Optional[str], schema: Optional[str], options: Optional[Mapping[str, str]], paths: Optional[List[str]], predicates: Optional[List[str]], is_streaming: Optional[bool])` |
| `sql.connect.plan.DataType` | Class | `(...)` |
| `sql.connect.plan.DatabaseExists` | Class | `(db_name: str)` |
| `sql.connect.plan.Deduplicate` | Class | `(child: Optional[LogicalPlan], all_columns_as_keys: bool, column_names: Optional[List[str]], within_watermark: bool)` |
| `sql.connect.plan.Drop` | Class | `(child: Optional[LogicalPlan], columns: List[Union[Column, str]])` |
| `sql.connect.plan.DropGlobalTempView` | Class | `(view_name: str)` |
| `sql.connect.plan.DropTempView` | Class | `(view_name: str)` |
| `sql.connect.plan.Expression` | Class | `()` |
| `sql.connect.plan.Filter` | Class | `(child: Optional[LogicalPlan], filter: Column)` |
| `sql.connect.plan.FunctionExists` | Class | `(function_name: str, db_name: Optional[str])` |
| `sql.connect.plan.GetDatabase` | Class | `(db_name: str)` |
| `sql.connect.plan.GetFunction` | Class | `(function_name: str, db_name: Optional[str])` |
| `sql.connect.plan.GetTable` | Class | `(table_name: str, db_name: Optional[str])` |
| `sql.connect.plan.GroupMap` | Class | `(child: Optional[LogicalPlan], grouping_cols: Sequence[Column], function: UserDefinedFunction, cols: List[str])` |
| `sql.connect.plan.Hint` | Class | `(child: Optional[LogicalPlan], name: str, parameters: Sequence[Column])` |
| `sql.connect.plan.HtmlString` | Class | `(child: Optional[LogicalPlan], num_rows: int, truncate: int)` |
| `sql.connect.plan.IsCached` | Class | `(table_name: str)` |
| `sql.connect.plan.Join` | Class | `(left: Optional[LogicalPlan], right: LogicalPlan, on: Optional[Union[str, List[str], Column, List[Column]]], how: Optional[str])` |
| `sql.connect.plan.LateralJoin` | Class | `(left: Optional[LogicalPlan], right: LogicalPlan, on: Optional[Column], how: Optional[str])` |
| `sql.connect.plan.Limit` | Class | `(child: Optional[LogicalPlan], limit: int)` |
| `sql.connect.plan.ListCatalogs` | Class | `(pattern: Optional[str])` |
| `sql.connect.plan.ListColumns` | Class | `(table_name: str, db_name: Optional[str])` |
| `sql.connect.plan.ListDatabases` | Class | `(pattern: Optional[str])` |
| `sql.connect.plan.ListFunctions` | Class | `(db_name: Optional[str], pattern: Optional[str])` |
| `sql.connect.plan.ListTables` | Class | `(db_name: Optional[str], pattern: Optional[str])` |
| `sql.connect.plan.LocalRelation` | Class | `(table: Optional[pa.Table], schema: Optional[str])` |
| `sql.connect.plan.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `sql.connect.plan.MapPartitions` | Class | `(child: Optional[LogicalPlan], function: UserDefinedFunction, cols: List[str], is_barrier: bool, profile: Optional[ResourceProfile])` |
| `sql.connect.plan.NADrop` | Class | `(child: Optional[LogicalPlan], cols: Optional[List[str]], min_non_nulls: Optional[int])` |
| `sql.connect.plan.NAFill` | Class | `(child: Optional[LogicalPlan], cols: Optional[List[str]], values: List[Any])` |
| `sql.connect.plan.NAReplace` | Class | `(child: Optional[LogicalPlan], cols: Optional[List[str]], replacements: Sequence[Tuple[Column, Column]])` |
| `sql.connect.plan.Observation` | Class | `(name: Optional[str])` |
| `sql.connect.plan.Offset` | Class | `(child: Optional[LogicalPlan], offset: int)` |
| `sql.connect.plan.Project` | Class | `(child: Optional[LogicalPlan], columns: List[Column])` |
| `sql.connect.plan.PySparkPicklingError` | Class | `(...)` |
| `sql.connect.plan.PySparkValueError` | Class | `(...)` |
| `sql.connect.plan.PythonDataSource` | Class | `(data_source: Type, python_ver: str)` |
| `sql.connect.plan.PythonUDTF` | Class | `(func: Type, return_type: Optional[Union[DataType, str]], eval_type: int, python_ver: str)` |
| `sql.connect.plan.Range` | Class | `(start: int, end: int, step: int, num_partitions: Optional[int])` |
| `sql.connect.plan.Read` | Class | `(table_name: str, options: Optional[Dict[str, str]], is_streaming: Optional[bool])` |
| `sql.connect.plan.RecoverPartitions` | Class | `(table_name: str)` |
| `sql.connect.plan.RefreshByPath` | Class | `(path: str)` |
| `sql.connect.plan.RefreshTable` | Class | `(table_name: str)` |
| `sql.connect.plan.RemoveRemoteCachedRelation` | Class | `(relation: CachedRemoteRelation)` |
| `sql.connect.plan.Repartition` | Class | `(child: Optional[LogicalPlan], num_partitions: int, shuffle: bool)` |
| `sql.connect.plan.RepartitionByExpression` | Class | `(child: Optional[LogicalPlan], num_partitions: Optional[int], columns: List[Column])` |
| `sql.connect.plan.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `sql.connect.plan.SQL` | Class | `(query: str, args: Optional[List[Column]], named_args: Optional[Dict[str, Column]], views: Optional[Sequence[SubqueryAlias]])` |
| `sql.connect.plan.Sample` | Class | `(child: Optional[LogicalPlan], lower_bound: float, upper_bound: float, with_replacement: bool, seed: int, deterministic_order: bool)` |
| `sql.connect.plan.SetCurrentCatalog` | Class | `(catalog_name: str)` |
| `sql.connect.plan.SetCurrentDatabase` | Class | `(db_name: str)` |
| `sql.connect.plan.SetOperation` | Class | `(child: Optional[LogicalPlan], other: Optional[LogicalPlan], set_op: str, is_all: bool, by_name: bool, allow_missing_columns: bool)` |
| `sql.connect.plan.ShowString` | Class | `(child: Optional[LogicalPlan], num_rows: int, truncate: int, vertical: bool)` |
| `sql.connect.plan.Sort` | Class | `(child: Optional[LogicalPlan], columns: List[Column], is_global: bool)` |
| `sql.connect.plan.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.plan.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.plan.StatApproxQuantile` | Class | `(child: Optional[LogicalPlan], cols: List[str], probabilities: List[float], relativeError: float)` |
| `sql.connect.plan.StatCorr` | Class | `(child: Optional[LogicalPlan], col1: str, col2: str, method: str)` |
| `sql.connect.plan.StatCov` | Class | `(child: Optional[LogicalPlan], col1: str, col2: str)` |
| `sql.connect.plan.StatCrosstab` | Class | `(child: Optional[LogicalPlan], col1: str, col2: str)` |
| `sql.connect.plan.StatDescribe` | Class | `(child: Optional[LogicalPlan], cols: List[str])` |
| `sql.connect.plan.StatFreqItems` | Class | `(child: Optional[LogicalPlan], cols: List[str], support: float)` |
| `sql.connect.plan.StatSampleBy` | Class | `(child: Optional[LogicalPlan], col: Column, fractions: Sequence[Tuple[Column, float]], seed: int)` |
| `sql.connect.plan.StatSummary` | Class | `(child: Optional[LogicalPlan], statistics: List[str])` |
| `sql.connect.plan.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.connect.plan.SubqueryAlias` | Class | `(child: Optional[LogicalPlan], alias: str)` |
| `sql.connect.plan.SubqueryExpression` | Class | `(plan: LogicalPlan, subquery_type: str, partition_spec: Optional[Sequence[Expression]], order_spec: Optional[Sequence[SortOrder]], with_single_partition: Optional[bool], in_subquery_values: Optional[Sequence[Expression]])` |
| `sql.connect.plan.TableExists` | Class | `(table_name: str, db_name: Optional[str])` |
| `sql.connect.plan.Tail` | Class | `(child: Optional[LogicalPlan], limit: int)` |
| `sql.connect.plan.ToDF` | Class | `(child: Optional[LogicalPlan], cols: Sequence[str])` |
| `sql.connect.plan.ToSchema` | Class | `(child: Optional[LogicalPlan], schema: DataType)` |
| `sql.connect.plan.TransformWithStateInPandas` | Class | `(...)` |
| `sql.connect.plan.TransformWithStateInPySpark` | Class | `(...)` |
| `sql.connect.plan.Transpose` | Class | `(child: Optional[LogicalPlan], index_columns: Sequence[Column])` |
| `sql.connect.plan.UncacheTable` | Class | `(table_name: str)` |
| `sql.connect.plan.UnparsedDataType` | Class | `(data_type_string: str)` |
| `sql.connect.plan.Unpivot` | Class | `(child: Optional[LogicalPlan], ids: List[Column], values: Optional[List[Column]], variable_column_name: str, value_column_name: str)` |
| `sql.connect.plan.UnresolvedTableValuedFunction` | Class | `(name: str, args: Sequence[Column])` |
| `sql.connect.plan.UserDefinedFunction` | Class | `(func: Callable[..., Any], returnType: DataTypeOrString, name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.connect.plan.WithColumns` | Class | `(child: Optional[LogicalPlan], columnNames: Sequence[str], columns: Sequence[Column], metadata: Optional[Sequence[str]])` |
| `sql.connect.plan.WithColumnsRenamed` | Class | `(child: Optional[LogicalPlan], colsMap: Mapping[str, str])` |
| `sql.connect.plan.WithRelations` | Class | `(child: Optional[LogicalPlan], references: Sequence[LogicalPlan])` |
| `sql.connect.plan.WithWatermark` | Class | `(child: Optional[LogicalPlan], event_time: str, delay_threshold: str)` |
| `sql.connect.plan.WriteOperation` | Class | `(child: LogicalPlan)` |
| `sql.connect.plan.WriteOperationV2` | Class | `(child: LogicalPlan, table_name: str)` |
| `sql.connect.plan.WriteStreamOperation` | Class | `(child: LogicalPlan)` |
| `sql.connect.plan.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.plan.logger` | Object | `` |
| `sql.connect.plan.proto` | Object | `` |
| `sql.connect.plan.pyspark_types_to_proto_types` | Function | `(data_type: DataType) -> pb2.DataType` |
| `sql.connect.plan.spark_dot_connect_dot_base__pb2` | Object | `` |
| `sql.connect.plan.storage_level_to_proto` | Function | `(storage_level: StorageLevel) -> pb2.StorageLevel` |
| `sql.connect.profiler.ConnectProfilerCollector` | Class | `()` |
| `sql.connect.profiler.ProfileResults` | Object | `` |
| `sql.connect.profiler.ProfileResultsParam` | Object | `` |
| `sql.connect.profiler.ProfilerCollector` | Class | `()` |
| `sql.connect.proto.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.SparkConnectService` | Class | `(...)` |
| `sql.connect.proto.SparkConnectServiceServicer` | Class | `(...)` |
| `sql.connect.proto.SparkConnectServiceStub` | Class | `(channel)` |
| `sql.connect.proto.add_SparkConnectServiceServicer_to_server` | Function | `(servicer, server)` |
| `sql.connect.proto.base_pb2.AddArtifactsRequest` | Class | `(session_id: builtins.str, user_context: global___UserContext \| None, client_observed_server_side_session_id: builtins.str \| None, client_type: builtins.str \| None, batch: global___AddArtifactsRequest.Batch \| None, begin_chunk: global___AddArtifactsRequest.BeginChunkedArtifact \| None, chunk: global___AddArtifactsRequest.ArtifactChunk \| None)` |
| `sql.connect.proto.base_pb2.AddArtifactsResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, artifacts: collections.abc.Iterable[global___AddArtifactsResponse.ArtifactSummary] \| None)` |
| `sql.connect.proto.base_pb2.AnalyzePlanRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, client_type: builtins.str \| None, schema: global___AnalyzePlanRequest.Schema \| None, explain: global___AnalyzePlanRequest.Explain \| None, tree_string: global___AnalyzePlanRequest.TreeString \| None, is_local: global___AnalyzePlanRequest.IsLocal \| None, is_streaming: global___AnalyzePlanRequest.IsStreaming \| None, input_files: global___AnalyzePlanRequest.InputFiles \| None, spark_version: global___AnalyzePlanRequest.SparkVersion \| None, ddl_parse: global___AnalyzePlanRequest.DDLParse \| None, same_semantics: global___AnalyzePlanRequest.SameSemantics \| None, semantic_hash: global___AnalyzePlanRequest.SemanticHash \| None, persist: global___AnalyzePlanRequest.Persist \| None, unpersist: global___AnalyzePlanRequest.Unpersist \| None, get_storage_level: global___AnalyzePlanRequest.GetStorageLevel \| None, json_to_ddl: global___AnalyzePlanRequest.JsonToDDL \| None)` |
| `sql.connect.proto.base_pb2.AnalyzePlanResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, schema: global___AnalyzePlanResponse.Schema \| None, explain: global___AnalyzePlanResponse.Explain \| None, tree_string: global___AnalyzePlanResponse.TreeString \| None, is_local: global___AnalyzePlanResponse.IsLocal \| None, is_streaming: global___AnalyzePlanResponse.IsStreaming \| None, input_files: global___AnalyzePlanResponse.InputFiles \| None, spark_version: global___AnalyzePlanResponse.SparkVersion \| None, ddl_parse: global___AnalyzePlanResponse.DDLParse \| None, same_semantics: global___AnalyzePlanResponse.SameSemantics \| None, semantic_hash: global___AnalyzePlanResponse.SemanticHash \| None, persist: global___AnalyzePlanResponse.Persist \| None, unpersist: global___AnalyzePlanResponse.Unpersist \| None, get_storage_level: global___AnalyzePlanResponse.GetStorageLevel \| None, json_to_ddl: global___AnalyzePlanResponse.JsonToDDL \| None)` |
| `sql.connect.proto.base_pb2.ArtifactStatusesRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, client_type: builtins.str \| None, names: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.base_pb2.ArtifactStatusesResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, statuses: collections.abc.Mapping[builtins.str, global___ArtifactStatusesResponse.ArtifactStatus] \| None)` |
| `sql.connect.proto.base_pb2.COMPRESSION_CODEC_UNSPECIFIED` | Object | `` |
| `sql.connect.proto.base_pb2.COMPRESSION_CODEC_ZSTD` | Object | `` |
| `sql.connect.proto.base_pb2.CheckpointCommandResult` | Class | `(relation: pyspark.sql.connect.proto.relations_pb2.CachedRemoteRelation \| None)` |
| `sql.connect.proto.base_pb2.CloneSessionRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, client_type: builtins.str \| None, new_session_id: builtins.str \| None)` |
| `sql.connect.proto.base_pb2.CloneSessionResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, new_session_id: builtins.str, new_server_side_session_id: builtins.str)` |
| `sql.connect.proto.base_pb2.CompressionCodec` | Class | `(...)` |
| `sql.connect.proto.base_pb2.ConfigRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, operation: global___ConfigRequest.Operation \| None, client_type: builtins.str \| None)` |
| `sql.connect.proto.base_pb2.ConfigResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, pairs: collections.abc.Iterable[global___KeyValue] \| None, warnings: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.base_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.base_pb2.ExecutePlanRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, operation_id: builtins.str \| None, plan: global___Plan \| None, client_type: builtins.str \| None, request_options: collections.abc.Iterable[global___ExecutePlanRequest.RequestOption] \| None, tags: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.base_pb2.ExecutePlanResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, operation_id: builtins.str, response_id: builtins.str, arrow_batch: global___ExecutePlanResponse.ArrowBatch \| None, sql_command_result: global___ExecutePlanResponse.SqlCommandResult \| None, write_stream_operation_start_result: pyspark.sql.connect.proto.commands_pb2.WriteStreamOperationStartResult \| None, streaming_query_command_result: pyspark.sql.connect.proto.commands_pb2.StreamingQueryCommandResult \| None, get_resources_command_result: pyspark.sql.connect.proto.commands_pb2.GetResourcesCommandResult \| None, streaming_query_manager_command_result: pyspark.sql.connect.proto.commands_pb2.StreamingQueryManagerCommandResult \| None, streaming_query_listener_events_result: pyspark.sql.connect.proto.commands_pb2.StreamingQueryListenerEventsResult \| None, result_complete: global___ExecutePlanResponse.ResultComplete \| None, create_resource_profile_command_result: pyspark.sql.connect.proto.commands_pb2.CreateResourceProfileCommandResult \| None, execution_progress: global___ExecutePlanResponse.ExecutionProgress \| None, checkpoint_command_result: global___CheckpointCommandResult \| None, ml_command_result: pyspark.sql.connect.proto.ml_pb2.MlCommandResult \| None, pipeline_event_result: pyspark.sql.connect.proto.pipelines_pb2.PipelineEventResult \| None, pipeline_command_result: pyspark.sql.connect.proto.pipelines_pb2.PipelineCommandResult \| None, pipeline_query_function_execution_signal: pyspark.sql.connect.proto.pipelines_pb2.PipelineQueryFunctionExecutionSignal \| None, extension: google.protobuf.any_pb2.Any \| None, metrics: global___ExecutePlanResponse.Metrics \| None, observed_metrics: collections.abc.Iterable[global___ExecutePlanResponse.ObservedMetrics] \| None, schema: pyspark.sql.connect.proto.types_pb2.DataType \| None)` |
| `sql.connect.proto.base_pb2.FetchErrorDetailsRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, error_id: builtins.str, client_type: builtins.str \| None)` |
| `sql.connect.proto.base_pb2.FetchErrorDetailsResponse` | Class | `(server_side_session_id: builtins.str, session_id: builtins.str, root_error_idx: builtins.int \| None, errors: collections.abc.Iterable[global___FetchErrorDetailsResponse.Error] \| None)` |
| `sql.connect.proto.base_pb2.InterruptRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, client_type: builtins.str \| None, interrupt_type: global___InterruptRequest.InterruptType.ValueType, operation_tag: builtins.str, operation_id: builtins.str)` |
| `sql.connect.proto.base_pb2.InterruptResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, interrupted_ids: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.base_pb2.KeyValue` | Class | `(key: builtins.str, value: builtins.str \| None)` |
| `sql.connect.proto.base_pb2.Plan` | Class | `(root: pyspark.sql.connect.proto.relations_pb2.Relation \| None, command: pyspark.sql.connect.proto.commands_pb2.Command \| None, compressed_operation: global___Plan.CompressedOperation \| None)` |
| `sql.connect.proto.base_pb2.ReattachExecuteRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, operation_id: builtins.str, client_type: builtins.str \| None, last_response_id: builtins.str \| None)` |
| `sql.connect.proto.base_pb2.ReattachOptions` | Class | `(reattachable: builtins.bool)` |
| `sql.connect.proto.base_pb2.ReleaseExecuteRequest` | Class | `(session_id: builtins.str, client_observed_server_side_session_id: builtins.str \| None, user_context: global___UserContext \| None, operation_id: builtins.str, client_type: builtins.str \| None, release_all: global___ReleaseExecuteRequest.ReleaseAll \| None, release_until: global___ReleaseExecuteRequest.ReleaseUntil \| None)` |
| `sql.connect.proto.base_pb2.ReleaseExecuteResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str, operation_id: builtins.str \| None)` |
| `sql.connect.proto.base_pb2.ReleaseSessionRequest` | Class | `(session_id: builtins.str, user_context: global___UserContext \| None, client_type: builtins.str \| None, allow_reconnect: builtins.bool)` |
| `sql.connect.proto.base_pb2.ReleaseSessionResponse` | Class | `(session_id: builtins.str, server_side_session_id: builtins.str)` |
| `sql.connect.proto.base_pb2.ResultChunkingOptions` | Class | `(allow_arrow_batch_chunking: builtins.bool, preferred_arrow_chunk_size: builtins.int \| None)` |
| `sql.connect.proto.base_pb2.UserContext` | Class | `(user_id: builtins.str, user_name: builtins.str, extensions: collections.abc.Iterable[google.protobuf.any_pb2.Any] \| None)` |
| `sql.connect.proto.base_pb2.global___AddArtifactsRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___AddArtifactsResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___AnalyzePlanRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___AnalyzePlanResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___ArtifactStatusesRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___ArtifactStatusesResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___CheckpointCommandResult` | Object | `` |
| `sql.connect.proto.base_pb2.global___CloneSessionRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___CloneSessionResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___CompressionCodec` | Object | `` |
| `sql.connect.proto.base_pb2.global___ConfigRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___ConfigResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___ExecutePlanRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___ExecutePlanResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___FetchErrorDetailsRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___FetchErrorDetailsResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___InterruptRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___InterruptResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___KeyValue` | Object | `` |
| `sql.connect.proto.base_pb2.global___Plan` | Object | `` |
| `sql.connect.proto.base_pb2.global___ReattachExecuteRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___ReattachOptions` | Object | `` |
| `sql.connect.proto.base_pb2.global___ReleaseExecuteRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___ReleaseExecuteResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___ReleaseSessionRequest` | Object | `` |
| `sql.connect.proto.base_pb2.global___ReleaseSessionResponse` | Object | `` |
| `sql.connect.proto.base_pb2.global___ResultChunkingOptions` | Object | `` |
| `sql.connect.proto.base_pb2.global___UserContext` | Object | `` |
| `sql.connect.proto.base_pb2.spark_dot_connect_dot_commands__pb2` | Object | `` |
| `sql.connect.proto.base_pb2.spark_dot_connect_dot_common__pb2` | Object | `` |
| `sql.connect.proto.base_pb2.spark_dot_connect_dot_expressions__pb2` | Object | `` |
| `sql.connect.proto.base_pb2.spark_dot_connect_dot_ml__pb2` | Object | `` |
| `sql.connect.proto.base_pb2.spark_dot_connect_dot_pipelines__pb2` | Object | `` |
| `sql.connect.proto.base_pb2.spark_dot_connect_dot_relations__pb2` | Object | `` |
| `sql.connect.proto.base_pb2.spark_dot_connect_dot_types__pb2` | Object | `` |
| `sql.connect.proto.base_pb2_grpc.SparkConnectService` | Class | `(...)` |
| `sql.connect.proto.base_pb2_grpc.SparkConnectServiceServicer` | Class | `(...)` |
| `sql.connect.proto.base_pb2_grpc.SparkConnectServiceStub` | Class | `(channel)` |
| `sql.connect.proto.base_pb2_grpc.add_SparkConnectServiceServicer_to_server` | Function | `(servicer, server)` |
| `sql.connect.proto.base_pb2_grpc.spark_dot_connect_dot_base__pb2` | Object | `` |
| `sql.connect.proto.catalog_pb2.CacheTable` | Class | `(table_name: builtins.str, storage_level: pyspark.sql.connect.proto.common_pb2.StorageLevel \| None)` |
| `sql.connect.proto.catalog_pb2.Catalog` | Class | `(current_database: global___CurrentDatabase \| None, set_current_database: global___SetCurrentDatabase \| None, list_databases: global___ListDatabases \| None, list_tables: global___ListTables \| None, list_functions: global___ListFunctions \| None, list_columns: global___ListColumns \| None, get_database: global___GetDatabase \| None, get_table: global___GetTable \| None, get_function: global___GetFunction \| None, database_exists: global___DatabaseExists \| None, table_exists: global___TableExists \| None, function_exists: global___FunctionExists \| None, create_external_table: global___CreateExternalTable \| None, create_table: global___CreateTable \| None, drop_temp_view: global___DropTempView \| None, drop_global_temp_view: global___DropGlobalTempView \| None, recover_partitions: global___RecoverPartitions \| None, is_cached: global___IsCached \| None, cache_table: global___CacheTable \| None, uncache_table: global___UncacheTable \| None, clear_cache: global___ClearCache \| None, refresh_table: global___RefreshTable \| None, refresh_by_path: global___RefreshByPath \| None, current_catalog: global___CurrentCatalog \| None, set_current_catalog: global___SetCurrentCatalog \| None, list_catalogs: global___ListCatalogs \| None)` |
| `sql.connect.proto.catalog_pb2.ClearCache` | Class | `()` |
| `sql.connect.proto.catalog_pb2.CreateExternalTable` | Class | `(table_name: builtins.str, path: builtins.str \| None, source: builtins.str \| None, schema: pyspark.sql.connect.proto.types_pb2.DataType \| None, options: collections.abc.Mapping[builtins.str, builtins.str] \| None)` |
| `sql.connect.proto.catalog_pb2.CreateTable` | Class | `(table_name: builtins.str, path: builtins.str \| None, source: builtins.str \| None, description: builtins.str \| None, schema: pyspark.sql.connect.proto.types_pb2.DataType \| None, options: collections.abc.Mapping[builtins.str, builtins.str] \| None)` |
| `sql.connect.proto.catalog_pb2.CurrentCatalog` | Class | `()` |
| `sql.connect.proto.catalog_pb2.CurrentDatabase` | Class | `()` |
| `sql.connect.proto.catalog_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.catalog_pb2.DatabaseExists` | Class | `(db_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.DropGlobalTempView` | Class | `(view_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.DropTempView` | Class | `(view_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.FunctionExists` | Class | `(function_name: builtins.str, db_name: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.GetDatabase` | Class | `(db_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.GetFunction` | Class | `(function_name: builtins.str, db_name: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.GetTable` | Class | `(table_name: builtins.str, db_name: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.IsCached` | Class | `(table_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.ListCatalogs` | Class | `(pattern: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.ListColumns` | Class | `(table_name: builtins.str, db_name: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.ListDatabases` | Class | `(pattern: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.ListFunctions` | Class | `(db_name: builtins.str \| None, pattern: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.ListTables` | Class | `(db_name: builtins.str \| None, pattern: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.RecoverPartitions` | Class | `(table_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.RefreshByPath` | Class | `(path: builtins.str)` |
| `sql.connect.proto.catalog_pb2.RefreshTable` | Class | `(table_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.SetCurrentCatalog` | Class | `(catalog_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.SetCurrentDatabase` | Class | `(db_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.TableExists` | Class | `(table_name: builtins.str, db_name: builtins.str \| None)` |
| `sql.connect.proto.catalog_pb2.UncacheTable` | Class | `(table_name: builtins.str)` |
| `sql.connect.proto.catalog_pb2.global___CacheTable` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___Catalog` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___ClearCache` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___CreateExternalTable` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___CreateTable` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___CurrentCatalog` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___CurrentDatabase` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___DatabaseExists` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___DropGlobalTempView` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___DropTempView` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___FunctionExists` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___GetDatabase` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___GetFunction` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___GetTable` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___IsCached` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___ListCatalogs` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___ListColumns` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___ListDatabases` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___ListFunctions` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___ListTables` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___RecoverPartitions` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___RefreshByPath` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___RefreshTable` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___SetCurrentCatalog` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___SetCurrentDatabase` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___TableExists` | Object | `` |
| `sql.connect.proto.catalog_pb2.global___UncacheTable` | Object | `` |
| `sql.connect.proto.catalog_pb2.spark_dot_connect_dot_common__pb2` | Object | `` |
| `sql.connect.proto.catalog_pb2.spark_dot_connect_dot_types__pb2` | Object | `` |
| `sql.connect.proto.commands_pb2.CheckpointCommand` | Class | `(relation: pyspark.sql.connect.proto.relations_pb2.Relation \| None, local: builtins.bool, eager: builtins.bool, storage_level: pyspark.sql.connect.proto.common_pb2.StorageLevel \| None)` |
| `sql.connect.proto.commands_pb2.Command` | Class | `(register_function: pyspark.sql.connect.proto.expressions_pb2.CommonInlineUserDefinedFunction \| None, write_operation: global___WriteOperation \| None, create_dataframe_view: global___CreateDataFrameViewCommand \| None, write_operation_v2: global___WriteOperationV2 \| None, sql_command: global___SqlCommand \| None, write_stream_operation_start: global___WriteStreamOperationStart \| None, streaming_query_command: global___StreamingQueryCommand \| None, get_resources_command: global___GetResourcesCommand \| None, streaming_query_manager_command: global___StreamingQueryManagerCommand \| None, register_table_function: pyspark.sql.connect.proto.relations_pb2.CommonInlineUserDefinedTableFunction \| None, streaming_query_listener_bus_command: global___StreamingQueryListenerBusCommand \| None, register_data_source: pyspark.sql.connect.proto.relations_pb2.CommonInlineUserDefinedDataSource \| None, create_resource_profile_command: global___CreateResourceProfileCommand \| None, checkpoint_command: global___CheckpointCommand \| None, remove_cached_remote_relation_command: global___RemoveCachedRemoteRelationCommand \| None, merge_into_table_command: global___MergeIntoTableCommand \| None, ml_command: pyspark.sql.connect.proto.ml_pb2.MlCommand \| None, execute_external_command: global___ExecuteExternalCommand \| None, pipeline_command: pyspark.sql.connect.proto.pipelines_pb2.PipelineCommand \| None, extension: google.protobuf.any_pb2.Any \| None)` |
| `sql.connect.proto.commands_pb2.CreateDataFrameViewCommand` | Class | `(input: pyspark.sql.connect.proto.relations_pb2.Relation \| None, name: builtins.str, is_global: builtins.bool, replace: builtins.bool)` |
| `sql.connect.proto.commands_pb2.CreateResourceProfileCommand` | Class | `(profile: pyspark.sql.connect.proto.common_pb2.ResourceProfile \| None)` |
| `sql.connect.proto.commands_pb2.CreateResourceProfileCommandResult` | Class | `(profile_id: builtins.int)` |
| `sql.connect.proto.commands_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.commands_pb2.ExecuteExternalCommand` | Class | `(runner: builtins.str, command: builtins.str, options: collections.abc.Mapping[builtins.str, builtins.str] \| None)` |
| `sql.connect.proto.commands_pb2.GetResourcesCommand` | Class | `()` |
| `sql.connect.proto.commands_pb2.GetResourcesCommandResult` | Class | `(resources: collections.abc.Mapping[builtins.str, pyspark.sql.connect.proto.common_pb2.ResourceInformation] \| None)` |
| `sql.connect.proto.commands_pb2.MergeIntoTableCommand` | Class | `(target_table_name: builtins.str, source_table_plan: pyspark.sql.connect.proto.relations_pb2.Relation \| None, merge_condition: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, match_actions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, not_matched_actions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, not_matched_by_source_actions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, with_schema_evolution: builtins.bool)` |
| `sql.connect.proto.commands_pb2.QUERY_IDLE_EVENT` | Object | `` |
| `sql.connect.proto.commands_pb2.QUERY_PROGRESS_EVENT` | Object | `` |
| `sql.connect.proto.commands_pb2.QUERY_PROGRESS_UNSPECIFIED` | Object | `` |
| `sql.connect.proto.commands_pb2.QUERY_TERMINATED_EVENT` | Object | `` |
| `sql.connect.proto.commands_pb2.RemoveCachedRemoteRelationCommand` | Class | `(relation: pyspark.sql.connect.proto.relations_pb2.CachedRemoteRelation \| None)` |
| `sql.connect.proto.commands_pb2.SqlCommand` | Class | `(sql: builtins.str, args: collections.abc.Mapping[builtins.str, pyspark.sql.connect.proto.expressions_pb2.Expression.Literal] \| None, pos_args: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression.Literal] \| None, named_arguments: collections.abc.Mapping[builtins.str, pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, pos_arguments: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, input: pyspark.sql.connect.proto.relations_pb2.Relation \| None)` |
| `sql.connect.proto.commands_pb2.StreamingForeachFunction` | Class | `(python_function: pyspark.sql.connect.proto.expressions_pb2.PythonUDF \| None, scala_function: pyspark.sql.connect.proto.expressions_pb2.ScalarScalaUDF \| None)` |
| `sql.connect.proto.commands_pb2.StreamingQueryCommand` | Class | `(query_id: global___StreamingQueryInstanceId \| None, status: builtins.bool, last_progress: builtins.bool, recent_progress: builtins.bool, stop: builtins.bool, process_all_available: builtins.bool, explain: global___StreamingQueryCommand.ExplainCommand \| None, exception: builtins.bool, await_termination: global___StreamingQueryCommand.AwaitTerminationCommand \| None)` |
| `sql.connect.proto.commands_pb2.StreamingQueryCommandResult` | Class | `(query_id: global___StreamingQueryInstanceId \| None, status: global___StreamingQueryCommandResult.StatusResult \| None, recent_progress: global___StreamingQueryCommandResult.RecentProgressResult \| None, explain: global___StreamingQueryCommandResult.ExplainResult \| None, exception: global___StreamingQueryCommandResult.ExceptionResult \| None, await_termination: global___StreamingQueryCommandResult.AwaitTerminationResult \| None)` |
| `sql.connect.proto.commands_pb2.StreamingQueryEventType` | Class | `(...)` |
| `sql.connect.proto.commands_pb2.StreamingQueryInstanceId` | Class | `(id: builtins.str, run_id: builtins.str)` |
| `sql.connect.proto.commands_pb2.StreamingQueryListenerBusCommand` | Class | `(add_listener_bus_listener: builtins.bool, remove_listener_bus_listener: builtins.bool)` |
| `sql.connect.proto.commands_pb2.StreamingQueryListenerEvent` | Class | `(event_json: builtins.str, event_type: global___StreamingQueryEventType.ValueType)` |
| `sql.connect.proto.commands_pb2.StreamingQueryListenerEventsResult` | Class | `(events: collections.abc.Iterable[global___StreamingQueryListenerEvent] \| None, listener_bus_listener_added: builtins.bool \| None)` |
| `sql.connect.proto.commands_pb2.StreamingQueryManagerCommand` | Class | `(active: builtins.bool, get_query: builtins.str, await_any_termination: global___StreamingQueryManagerCommand.AwaitAnyTerminationCommand \| None, reset_terminated: builtins.bool, add_listener: global___StreamingQueryManagerCommand.StreamingQueryListenerCommand \| None, remove_listener: global___StreamingQueryManagerCommand.StreamingQueryListenerCommand \| None, list_listeners: builtins.bool)` |
| `sql.connect.proto.commands_pb2.StreamingQueryManagerCommandResult` | Class | `(active: global___StreamingQueryManagerCommandResult.ActiveResult \| None, query: global___StreamingQueryManagerCommandResult.StreamingQueryInstance \| None, await_any_termination: global___StreamingQueryManagerCommandResult.AwaitAnyTerminationResult \| None, reset_terminated: builtins.bool, add_listener: builtins.bool, remove_listener: builtins.bool, list_listeners: global___StreamingQueryManagerCommandResult.ListStreamingQueryListenerResult \| None)` |
| `sql.connect.proto.commands_pb2.WriteOperation` | Class | `(input: pyspark.sql.connect.proto.relations_pb2.Relation \| None, source: builtins.str \| None, path: builtins.str, table: global___WriteOperation.SaveTable \| None, mode: global___WriteOperation.SaveMode.ValueType, sort_column_names: collections.abc.Iterable[builtins.str] \| None, partitioning_columns: collections.abc.Iterable[builtins.str] \| None, bucket_by: global___WriteOperation.BucketBy \| None, options: collections.abc.Mapping[builtins.str, builtins.str] \| None, clustering_columns: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.commands_pb2.WriteOperationV2` | Class | `(input: pyspark.sql.connect.proto.relations_pb2.Relation \| None, table_name: builtins.str, provider: builtins.str \| None, partitioning_columns: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, options: collections.abc.Mapping[builtins.str, builtins.str] \| None, table_properties: collections.abc.Mapping[builtins.str, builtins.str] \| None, mode: global___WriteOperationV2.Mode.ValueType, overwrite_condition: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, clustering_columns: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.commands_pb2.WriteStreamOperationStart` | Class | `(input: pyspark.sql.connect.proto.relations_pb2.Relation \| None, format: builtins.str, options: collections.abc.Mapping[builtins.str, builtins.str] \| None, partitioning_column_names: collections.abc.Iterable[builtins.str] \| None, processing_time_interval: builtins.str, available_now: builtins.bool, once: builtins.bool, continuous_checkpoint_interval: builtins.str, output_mode: builtins.str, query_name: builtins.str, path: builtins.str, table_name: builtins.str, foreach_writer: global___StreamingForeachFunction \| None, foreach_batch: global___StreamingForeachFunction \| None, clustering_column_names: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.commands_pb2.WriteStreamOperationStartResult` | Class | `(query_id: global___StreamingQueryInstanceId \| None, name: builtins.str, query_started_event_json: builtins.str \| None)` |
| `sql.connect.proto.commands_pb2.global___CheckpointCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___Command` | Object | `` |
| `sql.connect.proto.commands_pb2.global___CreateDataFrameViewCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___CreateResourceProfileCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___CreateResourceProfileCommandResult` | Object | `` |
| `sql.connect.proto.commands_pb2.global___ExecuteExternalCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___GetResourcesCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___GetResourcesCommandResult` | Object | `` |
| `sql.connect.proto.commands_pb2.global___MergeIntoTableCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___RemoveCachedRemoteRelationCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___SqlCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingForeachFunction` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryCommandResult` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryEventType` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryInstanceId` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryListenerBusCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryListenerEvent` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryListenerEventsResult` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryManagerCommand` | Object | `` |
| `sql.connect.proto.commands_pb2.global___StreamingQueryManagerCommandResult` | Object | `` |
| `sql.connect.proto.commands_pb2.global___WriteOperation` | Object | `` |
| `sql.connect.proto.commands_pb2.global___WriteOperationV2` | Object | `` |
| `sql.connect.proto.commands_pb2.global___WriteStreamOperationStart` | Object | `` |
| `sql.connect.proto.commands_pb2.global___WriteStreamOperationStartResult` | Object | `` |
| `sql.connect.proto.commands_pb2.spark_dot_connect_dot_common__pb2` | Object | `` |
| `sql.connect.proto.commands_pb2.spark_dot_connect_dot_expressions__pb2` | Object | `` |
| `sql.connect.proto.commands_pb2.spark_dot_connect_dot_ml__pb2` | Object | `` |
| `sql.connect.proto.commands_pb2.spark_dot_connect_dot_pipelines__pb2` | Object | `` |
| `sql.connect.proto.commands_pb2.spark_dot_connect_dot_relations__pb2` | Object | `` |
| `sql.connect.proto.common_pb2.Bools` | Class | `(values: collections.abc.Iterable[builtins.bool] \| None)` |
| `sql.connect.proto.common_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.common_pb2.Doubles` | Class | `(values: collections.abc.Iterable[builtins.float] \| None)` |
| `sql.connect.proto.common_pb2.ExecutorResourceRequest` | Class | `(resource_name: builtins.str, amount: builtins.int, discovery_script: builtins.str \| None, vendor: builtins.str \| None)` |
| `sql.connect.proto.common_pb2.Floats` | Class | `(values: collections.abc.Iterable[builtins.float] \| None)` |
| `sql.connect.proto.common_pb2.Ints` | Class | `(values: collections.abc.Iterable[builtins.int] \| None)` |
| `sql.connect.proto.common_pb2.JvmOrigin` | Class | `(line: builtins.int \| None, start_position: builtins.int \| None, start_index: builtins.int \| None, stop_index: builtins.int \| None, sql_text: builtins.str \| None, object_type: builtins.str \| None, object_name: builtins.str \| None, stack_trace: collections.abc.Iterable[global___StackTraceElement] \| None)` |
| `sql.connect.proto.common_pb2.Longs` | Class | `(values: collections.abc.Iterable[builtins.int] \| None)` |
| `sql.connect.proto.common_pb2.Origin` | Class | `(python_origin: global___PythonOrigin \| None, jvm_origin: global___JvmOrigin \| None)` |
| `sql.connect.proto.common_pb2.PythonOrigin` | Class | `(fragment: builtins.str, call_site: builtins.str)` |
| `sql.connect.proto.common_pb2.ResolvedIdentifier` | Class | `(catalog_name: builtins.str, namespace: collections.abc.Iterable[builtins.str] \| None, table_name: builtins.str)` |
| `sql.connect.proto.common_pb2.ResourceInformation` | Class | `(name: builtins.str, addresses: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.common_pb2.ResourceProfile` | Class | `(executor_resources: collections.abc.Mapping[builtins.str, global___ExecutorResourceRequest] \| None, task_resources: collections.abc.Mapping[builtins.str, global___TaskResourceRequest] \| None)` |
| `sql.connect.proto.common_pb2.StackTraceElement` | Class | `(class_loader_name: builtins.str \| None, module_name: builtins.str \| None, module_version: builtins.str \| None, declaring_class: builtins.str, method_name: builtins.str, file_name: builtins.str \| None, line_number: builtins.int)` |
| `sql.connect.proto.common_pb2.StorageLevel` | Class | `(use_disk: builtins.bool, use_memory: builtins.bool, use_off_heap: builtins.bool, deserialized: builtins.bool, replication: builtins.int)` |
| `sql.connect.proto.common_pb2.Strings` | Class | `(values: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.common_pb2.TaskResourceRequest` | Class | `(resource_name: builtins.str, amount: builtins.float)` |
| `sql.connect.proto.common_pb2.global___Bools` | Object | `` |
| `sql.connect.proto.common_pb2.global___Doubles` | Object | `` |
| `sql.connect.proto.common_pb2.global___ExecutorResourceRequest` | Object | `` |
| `sql.connect.proto.common_pb2.global___Floats` | Object | `` |
| `sql.connect.proto.common_pb2.global___Ints` | Object | `` |
| `sql.connect.proto.common_pb2.global___JvmOrigin` | Object | `` |
| `sql.connect.proto.common_pb2.global___Longs` | Object | `` |
| `sql.connect.proto.common_pb2.global___Origin` | Object | `` |
| `sql.connect.proto.common_pb2.global___PythonOrigin` | Object | `` |
| `sql.connect.proto.common_pb2.global___ResolvedIdentifier` | Object | `` |
| `sql.connect.proto.common_pb2.global___ResourceInformation` | Object | `` |
| `sql.connect.proto.common_pb2.global___ResourceProfile` | Object | `` |
| `sql.connect.proto.common_pb2.global___StackTraceElement` | Object | `` |
| `sql.connect.proto.common_pb2.global___StorageLevel` | Object | `` |
| `sql.connect.proto.common_pb2.global___Strings` | Object | `` |
| `sql.connect.proto.common_pb2.global___TaskResourceRequest` | Object | `` |
| `sql.connect.proto.example_plugins_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.example_plugins_pb2.ExamplePluginCommand` | Class | `(custom_field: builtins.str)` |
| `sql.connect.proto.example_plugins_pb2.ExamplePluginExpression` | Class | `(child: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, custom_field: builtins.str)` |
| `sql.connect.proto.example_plugins_pb2.ExamplePluginRelation` | Class | `(input: pyspark.sql.connect.proto.relations_pb2.Relation \| None, custom_field: builtins.str)` |
| `sql.connect.proto.example_plugins_pb2.global___ExamplePluginCommand` | Object | `` |
| `sql.connect.proto.example_plugins_pb2.global___ExamplePluginExpression` | Object | `` |
| `sql.connect.proto.example_plugins_pb2.global___ExamplePluginRelation` | Object | `` |
| `sql.connect.proto.example_plugins_pb2.spark_dot_connect_dot_expressions__pb2` | Object | `` |
| `sql.connect.proto.example_plugins_pb2.spark_dot_connect_dot_relations__pb2` | Object | `` |
| `sql.connect.proto.expressions_pb2.CallFunction` | Class | `(function_name: builtins.str, arguments: collections.abc.Iterable[global___Expression] \| None)` |
| `sql.connect.proto.expressions_pb2.CommonInlineUserDefinedFunction` | Class | `(function_name: builtins.str, deterministic: builtins.bool, arguments: collections.abc.Iterable[global___Expression] \| None, python_udf: global___PythonUDF \| None, scalar_scala_udf: global___ScalarScalaUDF \| None, java_udf: global___JavaUDF \| None, is_distinct: builtins.bool)` |
| `sql.connect.proto.expressions_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.expressions_pb2.Expression` | Class | `(common: global___ExpressionCommon \| None, literal: global___Expression.Literal \| None, unresolved_attribute: global___Expression.UnresolvedAttribute \| None, unresolved_function: global___Expression.UnresolvedFunction \| None, expression_string: global___Expression.ExpressionString \| None, unresolved_star: global___Expression.UnresolvedStar \| None, alias: global___Expression.Alias \| None, cast: global___Expression.Cast \| None, unresolved_regex: global___Expression.UnresolvedRegex \| None, sort_order: global___Expression.SortOrder \| None, lambda_function: global___Expression.LambdaFunction \| None, window: global___Expression.Window \| None, unresolved_extract_value: global___Expression.UnresolvedExtractValue \| None, update_fields: global___Expression.UpdateFields \| None, unresolved_named_lambda_variable: global___Expression.UnresolvedNamedLambdaVariable \| None, common_inline_user_defined_function: global___CommonInlineUserDefinedFunction \| None, call_function: global___CallFunction \| None, named_argument_expression: global___NamedArgumentExpression \| None, merge_action: global___MergeAction \| None, typed_aggregate_expression: global___TypedAggregateExpression \| None, subquery_expression: global___SubqueryExpression \| None, direct_shuffle_partition_id: global___Expression.DirectShufflePartitionID \| None, extension: google.protobuf.any_pb2.Any \| None)` |
| `sql.connect.proto.expressions_pb2.ExpressionCommon` | Class | `(origin: pyspark.sql.connect.proto.common_pb2.Origin \| None)` |
| `sql.connect.proto.expressions_pb2.JavaUDF` | Class | `(class_name: builtins.str, output_type: pyspark.sql.connect.proto.types_pb2.DataType \| None, aggregate: builtins.bool)` |
| `sql.connect.proto.expressions_pb2.MergeAction` | Class | `(action_type: global___MergeAction.ActionType.ValueType, condition: global___Expression \| None, assignments: collections.abc.Iterable[global___MergeAction.Assignment] \| None)` |
| `sql.connect.proto.expressions_pb2.NamedArgumentExpression` | Class | `(key: builtins.str, value: global___Expression \| None)` |
| `sql.connect.proto.expressions_pb2.PythonUDF` | Class | `(output_type: pyspark.sql.connect.proto.types_pb2.DataType \| None, eval_type: builtins.int, command: builtins.bytes, python_ver: builtins.str, additional_includes: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.expressions_pb2.ScalarScalaUDF` | Class | `(payload: builtins.bytes, inputTypes: collections.abc.Iterable[pyspark.sql.connect.proto.types_pb2.DataType] \| None, outputType: pyspark.sql.connect.proto.types_pb2.DataType \| None, nullable: builtins.bool, aggregate: builtins.bool)` |
| `sql.connect.proto.expressions_pb2.SubqueryExpression` | Class | `(plan_id: builtins.int, subquery_type: global___SubqueryExpression.SubqueryType.ValueType, table_arg_options: global___SubqueryExpression.TableArgOptions \| None, in_subquery_values: collections.abc.Iterable[global___Expression] \| None)` |
| `sql.connect.proto.expressions_pb2.TypedAggregateExpression` | Class | `(scalar_scala_udf: global___ScalarScalaUDF \| None)` |
| `sql.connect.proto.expressions_pb2.global___CallFunction` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___CommonInlineUserDefinedFunction` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___Expression` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___ExpressionCommon` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___JavaUDF` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___MergeAction` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___NamedArgumentExpression` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___PythonUDF` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___ScalarScalaUDF` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___SubqueryExpression` | Object | `` |
| `sql.connect.proto.expressions_pb2.global___TypedAggregateExpression` | Object | `` |
| `sql.connect.proto.expressions_pb2.spark_dot_connect_dot_common__pb2` | Object | `` |
| `sql.connect.proto.expressions_pb2.spark_dot_connect_dot_types__pb2` | Object | `` |
| `sql.connect.proto.google_dot_protobuf_dot_any__pb2` | Object | `` |
| `sql.connect.proto.google_dot_protobuf_dot_timestamp__pb2` | Object | `` |
| `sql.connect.proto.grpc` | Object | `` |
| `sql.connect.proto.ml_common_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.ml_common_pb2.MlOperator` | Class | `(name: builtins.str, uid: builtins.str, type: global___MlOperator.OperatorType.ValueType)` |
| `sql.connect.proto.ml_common_pb2.MlParams` | Class | `(params: collections.abc.Mapping[builtins.str, pyspark.sql.connect.proto.expressions_pb2.Expression.Literal] \| None)` |
| `sql.connect.proto.ml_common_pb2.ObjectRef` | Class | `(id: builtins.str)` |
| `sql.connect.proto.ml_common_pb2.global___MlOperator` | Object | `` |
| `sql.connect.proto.ml_common_pb2.global___MlParams` | Object | `` |
| `sql.connect.proto.ml_common_pb2.global___ObjectRef` | Object | `` |
| `sql.connect.proto.ml_common_pb2.spark_dot_connect_dot_expressions__pb2` | Object | `` |
| `sql.connect.proto.ml_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.ml_pb2.MlCommand` | Class | `(fit: global___MlCommand.Fit \| None, fetch: pyspark.sql.connect.proto.relations_pb2.Fetch \| None, delete: global___MlCommand.Delete \| None, write: global___MlCommand.Write \| None, read: global___MlCommand.Read \| None, evaluate: global___MlCommand.Evaluate \| None, clean_cache: global___MlCommand.CleanCache \| None, get_cache_info: global___MlCommand.GetCacheInfo \| None, create_summary: global___MlCommand.CreateSummary \| None, get_model_size: global___MlCommand.GetModelSize \| None)` |
| `sql.connect.proto.ml_pb2.MlCommandResult` | Class | `(param: pyspark.sql.connect.proto.expressions_pb2.Expression.Literal \| None, summary: builtins.str, operator_info: global___MlCommandResult.MlOperatorInfo \| None)` |
| `sql.connect.proto.ml_pb2.global___MlCommand` | Object | `` |
| `sql.connect.proto.ml_pb2.global___MlCommandResult` | Object | `` |
| `sql.connect.proto.ml_pb2.spark_dot_connect_dot_expressions__pb2` | Object | `` |
| `sql.connect.proto.ml_pb2.spark_dot_connect_dot_ml__common__pb2` | Object | `` |
| `sql.connect.proto.ml_pb2.spark_dot_connect_dot_relations__pb2` | Object | `` |
| `sql.connect.proto.pipelines_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.pipelines_pb2.MATERIALIZED_VIEW` | Object | `` |
| `sql.connect.proto.pipelines_pb2.OUTPUT_TYPE_UNSPECIFIED` | Object | `` |
| `sql.connect.proto.pipelines_pb2.OutputType` | Class | `(...)` |
| `sql.connect.proto.pipelines_pb2.PipelineAnalysisContext` | Class | `(dataflow_graph_id: builtins.str \| None, definition_path: builtins.str \| None, flow_name: builtins.str \| None, extension: collections.abc.Iterable[google.protobuf.any_pb2.Any] \| None)` |
| `sql.connect.proto.pipelines_pb2.PipelineCommand` | Class | `(create_dataflow_graph: global___PipelineCommand.CreateDataflowGraph \| None, define_output: global___PipelineCommand.DefineOutput \| None, define_flow: global___PipelineCommand.DefineFlow \| None, drop_dataflow_graph: global___PipelineCommand.DropDataflowGraph \| None, start_run: global___PipelineCommand.StartRun \| None, define_sql_graph_elements: global___PipelineCommand.DefineSqlGraphElements \| None, get_query_function_execution_signal_stream: global___PipelineCommand.GetQueryFunctionExecutionSignalStream \| None, define_flow_query_function_result: global___PipelineCommand.DefineFlowQueryFunctionResult \| None, extension: google.protobuf.any_pb2.Any \| None)` |
| `sql.connect.proto.pipelines_pb2.PipelineCommandResult` | Class | `(create_dataflow_graph_result: global___PipelineCommandResult.CreateDataflowGraphResult \| None, define_output_result: global___PipelineCommandResult.DefineOutputResult \| None, define_flow_result: global___PipelineCommandResult.DefineFlowResult \| None)` |
| `sql.connect.proto.pipelines_pb2.PipelineEvent` | Class | `(timestamp: google.protobuf.timestamp_pb2.Timestamp \| None, message: builtins.str \| None)` |
| `sql.connect.proto.pipelines_pb2.PipelineEventResult` | Class | `(event: global___PipelineEvent \| None)` |
| `sql.connect.proto.pipelines_pb2.PipelineQueryFunctionExecutionSignal` | Class | `(flow_names: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.pipelines_pb2.SINK` | Object | `` |
| `sql.connect.proto.pipelines_pb2.SourceCodeLocation` | Class | `(file_name: builtins.str \| None, line_number: builtins.int \| None, definition_path: builtins.str \| None, extension: collections.abc.Iterable[google.protobuf.any_pb2.Any] \| None)` |
| `sql.connect.proto.pipelines_pb2.TABLE` | Object | `` |
| `sql.connect.proto.pipelines_pb2.TEMPORARY_VIEW` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___OutputType` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___PipelineAnalysisContext` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___PipelineCommand` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___PipelineCommandResult` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___PipelineEvent` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___PipelineEventResult` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___PipelineQueryFunctionExecutionSignal` | Object | `` |
| `sql.connect.proto.pipelines_pb2.global___SourceCodeLocation` | Object | `` |
| `sql.connect.proto.pipelines_pb2.spark_dot_connect_dot_common__pb2` | Object | `` |
| `sql.connect.proto.pipelines_pb2.spark_dot_connect_dot_relations__pb2` | Object | `` |
| `sql.connect.proto.pipelines_pb2.spark_dot_connect_dot_types__pb2` | Object | `` |
| `sql.connect.proto.relations_pb2.Aggregate` | Class | `(input: global___Relation \| None, group_type: global___Aggregate.GroupType.ValueType, grouping_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, aggregate_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, pivot: global___Aggregate.Pivot \| None, grouping_sets: collections.abc.Iterable[global___Aggregate.GroupingSets] \| None)` |
| `sql.connect.proto.relations_pb2.ApplyInPandasWithState` | Class | `(input: global___Relation \| None, grouping_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, func: pyspark.sql.connect.proto.expressions_pb2.CommonInlineUserDefinedFunction \| None, output_schema: builtins.str, state_schema: builtins.str, output_mode: builtins.str, timeout_conf: builtins.str)` |
| `sql.connect.proto.relations_pb2.AsOfJoin` | Class | `(left: global___Relation \| None, right: global___Relation \| None, left_as_of: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, right_as_of: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, join_expr: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, using_columns: collections.abc.Iterable[builtins.str] \| None, join_type: builtins.str, tolerance: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, allow_exact_matches: builtins.bool, direction: builtins.str)` |
| `sql.connect.proto.relations_pb2.CachedLocalRelation` | Class | `(hash: builtins.str)` |
| `sql.connect.proto.relations_pb2.CachedRemoteRelation` | Class | `(relation_id: builtins.str)` |
| `sql.connect.proto.relations_pb2.ChunkedCachedLocalRelation` | Class | `(dataHashes: collections.abc.Iterable[builtins.str] \| None, schemaHash: builtins.str \| None)` |
| `sql.connect.proto.relations_pb2.CoGroupMap` | Class | `(input: global___Relation \| None, input_grouping_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, other: global___Relation \| None, other_grouping_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, func: pyspark.sql.connect.proto.expressions_pb2.CommonInlineUserDefinedFunction \| None, input_sorting_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, other_sorting_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None)` |
| `sql.connect.proto.relations_pb2.CollectMetrics` | Class | `(input: global___Relation \| None, name: builtins.str, metrics: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None)` |
| `sql.connect.proto.relations_pb2.CommonInlineUserDefinedDataSource` | Class | `(name: builtins.str, python_data_source: global___PythonDataSource \| None)` |
| `sql.connect.proto.relations_pb2.CommonInlineUserDefinedTableFunction` | Class | `(function_name: builtins.str, deterministic: builtins.bool, arguments: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, python_udtf: global___PythonUDTF \| None)` |
| `sql.connect.proto.relations_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.relations_pb2.Deduplicate` | Class | `(input: global___Relation \| None, column_names: collections.abc.Iterable[builtins.str] \| None, all_columns_as_keys: builtins.bool \| None, within_watermark: builtins.bool \| None)` |
| `sql.connect.proto.relations_pb2.Drop` | Class | `(input: global___Relation \| None, columns: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, column_names: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.relations_pb2.Fetch` | Class | `(obj_ref: pyspark.sql.connect.proto.ml_common_pb2.ObjectRef \| None, methods: collections.abc.Iterable[global___Fetch.Method] \| None)` |
| `sql.connect.proto.relations_pb2.Filter` | Class | `(input: global___Relation \| None, condition: pyspark.sql.connect.proto.expressions_pb2.Expression \| None)` |
| `sql.connect.proto.relations_pb2.GroupMap` | Class | `(input: global___Relation \| None, grouping_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, func: pyspark.sql.connect.proto.expressions_pb2.CommonInlineUserDefinedFunction \| None, sorting_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, initial_input: global___Relation \| None, initial_grouping_expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, is_map_groups_with_state: builtins.bool \| None, output_mode: builtins.str \| None, timeout_conf: builtins.str \| None, state_schema: pyspark.sql.connect.proto.types_pb2.DataType \| None, transform_with_state_info: global___TransformWithStateInfo \| None)` |
| `sql.connect.proto.relations_pb2.Hint` | Class | `(input: global___Relation \| None, name: builtins.str, parameters: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None)` |
| `sql.connect.proto.relations_pb2.HtmlString` | Class | `(input: global___Relation \| None, num_rows: builtins.int, truncate: builtins.int)` |
| `sql.connect.proto.relations_pb2.Join` | Class | `(left: global___Relation \| None, right: global___Relation \| None, join_condition: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, join_type: global___Join.JoinType.ValueType, using_columns: collections.abc.Iterable[builtins.str] \| None, join_data_type: global___Join.JoinDataType \| None)` |
| `sql.connect.proto.relations_pb2.LateralJoin` | Class | `(left: global___Relation \| None, right: global___Relation \| None, join_condition: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, join_type: global___Join.JoinType.ValueType)` |
| `sql.connect.proto.relations_pb2.Limit` | Class | `(input: global___Relation \| None, limit: builtins.int)` |
| `sql.connect.proto.relations_pb2.LocalRelation` | Class | `(data: builtins.bytes \| None, schema: builtins.str \| None)` |
| `sql.connect.proto.relations_pb2.MapPartitions` | Class | `(input: global___Relation \| None, func: pyspark.sql.connect.proto.expressions_pb2.CommonInlineUserDefinedFunction \| None, is_barrier: builtins.bool \| None, profile_id: builtins.int \| None)` |
| `sql.connect.proto.relations_pb2.MlRelation` | Class | `(transform: global___MlRelation.Transform \| None, fetch: global___Fetch \| None, model_summary_dataset: global___Relation \| None)` |
| `sql.connect.proto.relations_pb2.NADrop` | Class | `(input: global___Relation \| None, cols: collections.abc.Iterable[builtins.str] \| None, min_non_nulls: builtins.int \| None)` |
| `sql.connect.proto.relations_pb2.NAFill` | Class | `(input: global___Relation \| None, cols: collections.abc.Iterable[builtins.str] \| None, values: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression.Literal] \| None)` |
| `sql.connect.proto.relations_pb2.NAReplace` | Class | `(input: global___Relation \| None, cols: collections.abc.Iterable[builtins.str] \| None, replacements: collections.abc.Iterable[global___NAReplace.Replacement] \| None)` |
| `sql.connect.proto.relations_pb2.Offset` | Class | `(input: global___Relation \| None, offset: builtins.int)` |
| `sql.connect.proto.relations_pb2.Parse` | Class | `(input: global___Relation \| None, format: global___Parse.ParseFormat.ValueType, schema: pyspark.sql.connect.proto.types_pb2.DataType \| None, options: collections.abc.Mapping[builtins.str, builtins.str] \| None)` |
| `sql.connect.proto.relations_pb2.Project` | Class | `(input: global___Relation \| None, expressions: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None)` |
| `sql.connect.proto.relations_pb2.PythonDataSource` | Class | `(command: builtins.bytes, python_ver: builtins.str)` |
| `sql.connect.proto.relations_pb2.PythonUDTF` | Class | `(return_type: pyspark.sql.connect.proto.types_pb2.DataType \| None, eval_type: builtins.int, command: builtins.bytes, python_ver: builtins.str)` |
| `sql.connect.proto.relations_pb2.Range` | Class | `(start: builtins.int \| None, end: builtins.int, step: builtins.int, num_partitions: builtins.int \| None)` |
| `sql.connect.proto.relations_pb2.Read` | Class | `(named_table: global___Read.NamedTable \| None, data_source: global___Read.DataSource \| None, is_streaming: builtins.bool)` |
| `sql.connect.proto.relations_pb2.Relation` | Class | `(common: global___RelationCommon \| None, read: global___Read \| None, project: global___Project \| None, filter: global___Filter \| None, join: global___Join \| None, set_op: global___SetOperation \| None, sort: global___Sort \| None, limit: global___Limit \| None, aggregate: global___Aggregate \| None, sql: global___SQL \| None, local_relation: global___LocalRelation \| None, sample: global___Sample \| None, offset: global___Offset \| None, deduplicate: global___Deduplicate \| None, range: global___Range \| None, subquery_alias: global___SubqueryAlias \| None, repartition: global___Repartition \| None, to_df: global___ToDF \| None, with_columns_renamed: global___WithColumnsRenamed \| None, show_string: global___ShowString \| None, drop: global___Drop \| None, tail: global___Tail \| None, with_columns: global___WithColumns \| None, hint: global___Hint \| None, unpivot: global___Unpivot \| None, to_schema: global___ToSchema \| None, repartition_by_expression: global___RepartitionByExpression \| None, map_partitions: global___MapPartitions \| None, collect_metrics: global___CollectMetrics \| None, parse: global___Parse \| None, group_map: global___GroupMap \| None, co_group_map: global___CoGroupMap \| None, with_watermark: global___WithWatermark \| None, apply_in_pandas_with_state: global___ApplyInPandasWithState \| None, html_string: global___HtmlString \| None, cached_local_relation: global___CachedLocalRelation \| None, cached_remote_relation: global___CachedRemoteRelation \| None, common_inline_user_defined_table_function: global___CommonInlineUserDefinedTableFunction \| None, as_of_join: global___AsOfJoin \| None, common_inline_user_defined_data_source: global___CommonInlineUserDefinedDataSource \| None, with_relations: global___WithRelations \| None, transpose: global___Transpose \| None, unresolved_table_valued_function: global___UnresolvedTableValuedFunction \| None, lateral_join: global___LateralJoin \| None, chunked_cached_local_relation: global___ChunkedCachedLocalRelation \| None, fill_na: global___NAFill \| None, drop_na: global___NADrop \| None, replace: global___NAReplace \| None, summary: global___StatSummary \| None, crosstab: global___StatCrosstab \| None, describe: global___StatDescribe \| None, cov: global___StatCov \| None, corr: global___StatCorr \| None, approx_quantile: global___StatApproxQuantile \| None, freq_items: global___StatFreqItems \| None, sample_by: global___StatSampleBy \| None, catalog: pyspark.sql.connect.proto.catalog_pb2.Catalog \| None, ml_relation: global___MlRelation \| None, extension: google.protobuf.any_pb2.Any \| None, unknown: global___Unknown \| None)` |
| `sql.connect.proto.relations_pb2.RelationCommon` | Class | `(source_info: builtins.str, plan_id: builtins.int \| None, origin: pyspark.sql.connect.proto.common_pb2.Origin \| None)` |
| `sql.connect.proto.relations_pb2.Repartition` | Class | `(input: global___Relation \| None, num_partitions: builtins.int, shuffle: builtins.bool \| None)` |
| `sql.connect.proto.relations_pb2.RepartitionByExpression` | Class | `(input: global___Relation \| None, partition_exprs: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, num_partitions: builtins.int \| None)` |
| `sql.connect.proto.relations_pb2.SQL` | Class | `(query: builtins.str, args: collections.abc.Mapping[builtins.str, pyspark.sql.connect.proto.expressions_pb2.Expression.Literal] \| None, pos_args: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression.Literal] \| None, named_arguments: collections.abc.Mapping[builtins.str, pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, pos_arguments: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None)` |
| `sql.connect.proto.relations_pb2.Sample` | Class | `(input: global___Relation \| None, lower_bound: builtins.float, upper_bound: builtins.float, with_replacement: builtins.bool \| None, seed: builtins.int \| None, deterministic_order: builtins.bool)` |
| `sql.connect.proto.relations_pb2.SetOperation` | Class | `(left_input: global___Relation \| None, right_input: global___Relation \| None, set_op_type: global___SetOperation.SetOpType.ValueType, is_all: builtins.bool \| None, by_name: builtins.bool \| None, allow_missing_columns: builtins.bool \| None)` |
| `sql.connect.proto.relations_pb2.ShowString` | Class | `(input: global___Relation \| None, num_rows: builtins.int, truncate: builtins.int, vertical: builtins.bool)` |
| `sql.connect.proto.relations_pb2.Sort` | Class | `(input: global___Relation \| None, order: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression.SortOrder] \| None, is_global: builtins.bool \| None)` |
| `sql.connect.proto.relations_pb2.StatApproxQuantile` | Class | `(input: global___Relation \| None, cols: collections.abc.Iterable[builtins.str] \| None, probabilities: collections.abc.Iterable[builtins.float] \| None, relative_error: builtins.float)` |
| `sql.connect.proto.relations_pb2.StatCorr` | Class | `(input: global___Relation \| None, col1: builtins.str, col2: builtins.str, method: builtins.str \| None)` |
| `sql.connect.proto.relations_pb2.StatCov` | Class | `(input: global___Relation \| None, col1: builtins.str, col2: builtins.str)` |
| `sql.connect.proto.relations_pb2.StatCrosstab` | Class | `(input: global___Relation \| None, col1: builtins.str, col2: builtins.str)` |
| `sql.connect.proto.relations_pb2.StatDescribe` | Class | `(input: global___Relation \| None, cols: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.relations_pb2.StatFreqItems` | Class | `(input: global___Relation \| None, cols: collections.abc.Iterable[builtins.str] \| None, support: builtins.float \| None)` |
| `sql.connect.proto.relations_pb2.StatSampleBy` | Class | `(input: global___Relation \| None, col: pyspark.sql.connect.proto.expressions_pb2.Expression \| None, fractions: collections.abc.Iterable[global___StatSampleBy.Fraction] \| None, seed: builtins.int \| None)` |
| `sql.connect.proto.relations_pb2.StatSummary` | Class | `(input: global___Relation \| None, statistics: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.relations_pb2.SubqueryAlias` | Class | `(input: global___Relation \| None, alias: builtins.str, qualifier: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.relations_pb2.Tail` | Class | `(input: global___Relation \| None, limit: builtins.int)` |
| `sql.connect.proto.relations_pb2.ToDF` | Class | `(input: global___Relation \| None, column_names: collections.abc.Iterable[builtins.str] \| None)` |
| `sql.connect.proto.relations_pb2.ToSchema` | Class | `(input: global___Relation \| None, schema: pyspark.sql.connect.proto.types_pb2.DataType \| None)` |
| `sql.connect.proto.relations_pb2.TransformWithStateInfo` | Class | `(time_mode: builtins.str, event_time_column_name: builtins.str \| None, output_schema: pyspark.sql.connect.proto.types_pb2.DataType \| None)` |
| `sql.connect.proto.relations_pb2.Transpose` | Class | `(input: global___Relation \| None, index_columns: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None)` |
| `sql.connect.proto.relations_pb2.Unknown` | Class | `()` |
| `sql.connect.proto.relations_pb2.Unpivot` | Class | `(input: global___Relation \| None, ids: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None, values: global___Unpivot.Values \| None, variable_column_name: builtins.str, value_column_name: builtins.str)` |
| `sql.connect.proto.relations_pb2.UnresolvedTableValuedFunction` | Class | `(function_name: builtins.str, arguments: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression] \| None)` |
| `sql.connect.proto.relations_pb2.WithColumns` | Class | `(input: global___Relation \| None, aliases: collections.abc.Iterable[pyspark.sql.connect.proto.expressions_pb2.Expression.Alias] \| None)` |
| `sql.connect.proto.relations_pb2.WithColumnsRenamed` | Class | `(input: global___Relation \| None, rename_columns_map: collections.abc.Mapping[builtins.str, builtins.str] \| None, renames: collections.abc.Iterable[global___WithColumnsRenamed.Rename] \| None)` |
| `sql.connect.proto.relations_pb2.WithRelations` | Class | `(root: global___Relation \| None, references: collections.abc.Iterable[global___Relation] \| None)` |
| `sql.connect.proto.relations_pb2.WithWatermark` | Class | `(input: global___Relation \| None, event_time: builtins.str, delay_threshold: builtins.str)` |
| `sql.connect.proto.relations_pb2.global___Aggregate` | Object | `` |
| `sql.connect.proto.relations_pb2.global___ApplyInPandasWithState` | Object | `` |
| `sql.connect.proto.relations_pb2.global___AsOfJoin` | Object | `` |
| `sql.connect.proto.relations_pb2.global___CachedLocalRelation` | Object | `` |
| `sql.connect.proto.relations_pb2.global___CachedRemoteRelation` | Object | `` |
| `sql.connect.proto.relations_pb2.global___ChunkedCachedLocalRelation` | Object | `` |
| `sql.connect.proto.relations_pb2.global___CoGroupMap` | Object | `` |
| `sql.connect.proto.relations_pb2.global___CollectMetrics` | Object | `` |
| `sql.connect.proto.relations_pb2.global___CommonInlineUserDefinedDataSource` | Object | `` |
| `sql.connect.proto.relations_pb2.global___CommonInlineUserDefinedTableFunction` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Deduplicate` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Drop` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Fetch` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Filter` | Object | `` |
| `sql.connect.proto.relations_pb2.global___GroupMap` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Hint` | Object | `` |
| `sql.connect.proto.relations_pb2.global___HtmlString` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Join` | Object | `` |
| `sql.connect.proto.relations_pb2.global___LateralJoin` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Limit` | Object | `` |
| `sql.connect.proto.relations_pb2.global___LocalRelation` | Object | `` |
| `sql.connect.proto.relations_pb2.global___MapPartitions` | Object | `` |
| `sql.connect.proto.relations_pb2.global___MlRelation` | Object | `` |
| `sql.connect.proto.relations_pb2.global___NADrop` | Object | `` |
| `sql.connect.proto.relations_pb2.global___NAFill` | Object | `` |
| `sql.connect.proto.relations_pb2.global___NAReplace` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Offset` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Parse` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Project` | Object | `` |
| `sql.connect.proto.relations_pb2.global___PythonDataSource` | Object | `` |
| `sql.connect.proto.relations_pb2.global___PythonUDTF` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Range` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Read` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Relation` | Object | `` |
| `sql.connect.proto.relations_pb2.global___RelationCommon` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Repartition` | Object | `` |
| `sql.connect.proto.relations_pb2.global___RepartitionByExpression` | Object | `` |
| `sql.connect.proto.relations_pb2.global___SQL` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Sample` | Object | `` |
| `sql.connect.proto.relations_pb2.global___SetOperation` | Object | `` |
| `sql.connect.proto.relations_pb2.global___ShowString` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Sort` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatApproxQuantile` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatCorr` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatCov` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatCrosstab` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatDescribe` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatFreqItems` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatSampleBy` | Object | `` |
| `sql.connect.proto.relations_pb2.global___StatSummary` | Object | `` |
| `sql.connect.proto.relations_pb2.global___SubqueryAlias` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Tail` | Object | `` |
| `sql.connect.proto.relations_pb2.global___ToDF` | Object | `` |
| `sql.connect.proto.relations_pb2.global___ToSchema` | Object | `` |
| `sql.connect.proto.relations_pb2.global___TransformWithStateInfo` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Transpose` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Unknown` | Object | `` |
| `sql.connect.proto.relations_pb2.global___Unpivot` | Object | `` |
| `sql.connect.proto.relations_pb2.global___UnresolvedTableValuedFunction` | Object | `` |
| `sql.connect.proto.relations_pb2.global___WithColumns` | Object | `` |
| `sql.connect.proto.relations_pb2.global___WithColumnsRenamed` | Object | `` |
| `sql.connect.proto.relations_pb2.global___WithRelations` | Object | `` |
| `sql.connect.proto.relations_pb2.global___WithWatermark` | Object | `` |
| `sql.connect.proto.relations_pb2.spark_dot_connect_dot_catalog__pb2` | Object | `` |
| `sql.connect.proto.relations_pb2.spark_dot_connect_dot_common__pb2` | Object | `` |
| `sql.connect.proto.relations_pb2.spark_dot_connect_dot_expressions__pb2` | Object | `` |
| `sql.connect.proto.relations_pb2.spark_dot_connect_dot_ml__common__pb2` | Object | `` |
| `sql.connect.proto.relations_pb2.spark_dot_connect_dot_types__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_base__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_catalog__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_commands__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_common__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_expressions__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_ml__common__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_ml__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_pipelines__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_relations__pb2` | Object | `` |
| `sql.connect.proto.spark_dot_connect_dot_types__pb2` | Object | `` |
| `sql.connect.proto.types_pb2.DESCRIPTOR` | Object | `` |
| `sql.connect.proto.types_pb2.DataType` | Class | `(null: global___DataType.NULL \| None, binary: global___DataType.Binary \| None, boolean: global___DataType.Boolean \| None, byte: global___DataType.Byte \| None, short: global___DataType.Short \| None, integer: global___DataType.Integer \| None, long: global___DataType.Long \| None, float: global___DataType.Float \| None, double: global___DataType.Double \| None, decimal: global___DataType.Decimal \| None, string: global___DataType.String \| None, char: global___DataType.Char \| None, var_char: global___DataType.VarChar \| None, date: global___DataType.Date \| None, timestamp: global___DataType.Timestamp \| None, timestamp_ntz: global___DataType.TimestampNTZ \| None, calendar_interval: global___DataType.CalendarInterval \| None, year_month_interval: global___DataType.YearMonthInterval \| None, day_time_interval: global___DataType.DayTimeInterval \| None, array: global___DataType.Array \| None, struct: global___DataType.Struct \| None, map: global___DataType.Map \| None, variant: global___DataType.Variant \| None, udt: global___DataType.UDT \| None, geometry: global___DataType.Geometry \| None, geography: global___DataType.Geography \| None, unparsed: global___DataType.Unparsed \| None, time: global___DataType.Time \| None)` |
| `sql.connect.proto.types_pb2.global___DataType` | Object | `` |
| `sql.connect.protobuf.functions.Column` | Class | `(...)` |
| `sql.connect.protobuf.functions.ColumnOrName` | Object | `` |
| `sql.connect.protobuf.functions.PyProtobufFunctions` | Object | `` |
| `sql.connect.protobuf.functions.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.protobuf.functions.from_protobuf` | Function | `(data: ColumnOrName, messageName: str, descFilePath: Optional[str], options: Optional[Dict[str, str]], binaryDescriptorSet: Optional[bytes]) -> Column` |
| `sql.connect.protobuf.functions.lit` | Function | `(col: Any) -> Column` |
| `sql.connect.protobuf.functions.to_protobuf` | Function | `(data: ColumnOrName, messageName: str, descFilePath: Optional[str], options: Optional[Dict[str, str]], binaryDescriptorSet: Optional[bytes]) -> Column` |
| `sql.connect.readwriter.ColumnOrName` | Object | `` |
| `sql.connect.readwriter.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.readwriter.DataFrameReader` | Class | `(client: SparkSession)` |
| `sql.connect.readwriter.DataFrameWriter` | Class | `(plan: LogicalPlan, session: SparkSession, callback: Optional[Callable[[ExecutionInfo], None]])` |
| `sql.connect.readwriter.DataFrameWriterV2` | Class | `(plan: LogicalPlan, session: SparkSession, table: str, callback: Optional[Callable[[ExecutionInfo], None]])` |
| `sql.connect.readwriter.DataSource` | Class | `(format: Optional[str], schema: Optional[str], options: Optional[Mapping[str, str]], paths: Optional[List[str]], predicates: Optional[List[str]], is_streaming: Optional[bool])` |
| `sql.connect.readwriter.ExecutionInfo` | Class | `(metrics: Optional[list[PlanMetrics]], obs: Optional[Sequence[ObservedMetrics]])` |
| `sql.connect.readwriter.F` | Object | `` |
| `sql.connect.readwriter.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `sql.connect.readwriter.OptionUtils` | Class | `(...)` |
| `sql.connect.readwriter.OptionalPrimitiveType` | Object | `` |
| `sql.connect.readwriter.PathOrPaths` | Object | `` |
| `sql.connect.readwriter.PySparkAttributeError` | Class | `(...)` |
| `sql.connect.readwriter.PySparkDataFrameReader` | Class | `(spark: SparkSession)` |
| `sql.connect.readwriter.PySparkDataFrameWriter` | Class | `(df: DataFrame)` |
| `sql.connect.readwriter.PySparkDataFrameWriterV2` | Class | `(df: DataFrame, table: str)` |
| `sql.connect.readwriter.PySparkTypeError` | Class | `(...)` |
| `sql.connect.readwriter.PySparkValueError` | Class | `(...)` |
| `sql.connect.readwriter.Read` | Class | `(table_name: str, options: Optional[Dict[str, str]], is_streaming: Optional[bool])` |
| `sql.connect.readwriter.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.readwriter.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.readwriter.TupleOrListOfString` | Object | `` |
| `sql.connect.readwriter.WriteOperation` | Class | `(child: LogicalPlan)` |
| `sql.connect.readwriter.WriteOperationV2` | Class | `(child: LogicalPlan, table_name: str)` |
| `sql.connect.readwriter.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.readwriter.to_str` | Function | `(value: Any) -> Optional[str]` |
| `sql.connect.resource.profile.ExecutorResourceRequest` | Class | `(resourceName: str, amount: int, discoveryScript: str, vendor: str)` |
| `sql.connect.resource.profile.ResourceProfile` | Class | `(exec_req: Optional[Dict[str, ExecutorResourceRequest]], task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `sql.connect.resource.profile.TaskResourceRequest` | Class | `(resourceName: str, amount: float)` |
| `sql.connect.resource.profile.pb2` | Object | `` |
| `sql.connect.session.ArrowStreamPandasSerializer` | Class | `(timezone, safecheck, int_to_decimal_coercion_enabled)` |
| `sql.connect.session.AtomicType` | Class | `(...)` |
| `sql.connect.session.CachedRelation` | Class | `(plan: proto.Relation)` |
| `sql.connect.session.CachedRemoteRelation` | Class | `(relation_id: str, spark_session: SparkSession)` |
| `sql.connect.session.Catalog` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.session.ChunkedCachedLocalRelation` | Class | `(data_hashes: list[str], schema_hash: Optional[str])` |
| `sql.connect.session.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.session.DataFrameReader` | Class | `(client: SparkSession)` |
| `sql.connect.session.DataSourceRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.session.DataStreamReader` | Class | `(client: SparkSession)` |
| `sql.connect.session.DataType` | Class | `(...)` |
| `sql.connect.session.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `sql.connect.session.DefaultChannelBuilder` | Class | `(url: str, channelOptions: Optional[List[Tuple[str, Any]]])` |
| `sql.connect.session.F` | Object | `` |
| `sql.connect.session.LocalRelation` | Class | `(table: Optional[pa.Table], schema: Optional[str])` |
| `sql.connect.session.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `sql.connect.session.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `sql.connect.session.OptionalPrimitiveType` | Object | `` |
| `sql.connect.session.ParentDataFrame` | Class | `(...)` |
| `sql.connect.session.Profile` | Class | `(profiler_collector: ProfilerCollector)` |
| `sql.connect.session.ProfilerCollector` | Class | `()` |
| `sql.connect.session.ProgressHandler` | Class | `(...)` |
| `sql.connect.session.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.connect.session.PySparkAttributeError` | Class | `(...)` |
| `sql.connect.session.PySparkNotImplementedError` | Class | `(...)` |
| `sql.connect.session.PySparkRuntimeError` | Class | `(...)` |
| `sql.connect.session.PySparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.connect.session.PySparkTypeError` | Class | `(...)` |
| `sql.connect.session.PySparkValueError` | Class | `(...)` |
| `sql.connect.session.Range` | Class | `(start: int, end: int, step: int, num_partitions: Optional[int])` |
| `sql.connect.session.Row` | Class | `(...)` |
| `sql.connect.session.RuntimeConf` | Class | `(client: SparkConnectClient)` |
| `sql.connect.session.SQL` | Class | `(query: str, args: Optional[List[Column]], named_args: Optional[Dict[str, Column]], views: Optional[Sequence[SubqueryAlias]])` |
| `sql.connect.session.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.session.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.session.StreamingQueryManager` | Class | `(session: SparkSession)` |
| `sql.connect.session.StringType` | Class | `(collation: str)` |
| `sql.connect.session.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.session.SubqueryAlias` | Class | `(child: Optional[LogicalPlan], alias: str)` |
| `sql.connect.session.TableValuedFunction` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.session.TimestampType` | Class | `(...)` |
| `sql.connect.session.UDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.session.UDTFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.session.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.session.classproperty` | Class | `(...)` |
| `sql.connect.session.from_arrow_schema` | Function | `(arrow_schema: pa.Schema, prefer_timestamp_ntz: bool) -> StructType` |
| `sql.connect.session.from_arrow_type` | Function | `(at: pa.DataType, prefer_timestamp_ntz: bool) -> DataType` |
| `sql.connect.session.logger` | Object | `` |
| `sql.connect.session.pb2` | Object | `` |
| `sql.connect.session.to_arrow_schema` | Function | `(schema: StructType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.Schema` |
| `sql.connect.session.to_arrow_type` | Function | `(dt: DataType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.DataType` |
| `sql.connect.session.to_str` | Function | `(value: Any) -> Optional[str]` |
| `sql.connect.shell.PROGRESS_BAR_ENABLED` | Object | `` |
| `sql.connect.shell.progress.ExecutePlanResponse` | Object | `` |
| `sql.connect.shell.progress.Progress` | Class | `(char: str, min_width: int, output: typing.IO, enabled: bool, handlers: Iterable[ProgressHandler], operation_id: typing.Optional[str])` |
| `sql.connect.shell.progress.ProgressHandler` | Class | `(...)` |
| `sql.connect.shell.progress.StageInfo` | Class | `(stage_id: int, num_tasks: int, num_completed_tasks: int, num_bytes_read: int, done: bool)` |
| `sql.connect.shell.progress.from_proto` | Function | `(proto: ExecutePlanResponse.ExecutionProgress) -> typing.Tuple[Iterable[StageInfo], int]` |
| `sql.connect.shell.progress.get_terminal_size` | Function | `(defaultx: Any, defaulty: Any) -> Any` |
| `sql.connect.shell.progress.progress_bar_enabled` | Function | `() -> bool` |
| `sql.connect.shell.progress_bar_enabled` | Function | `() -> bool` |
| `sql.connect.sql_formatter.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.sql_formatter.PySparkValueError` | Class | `(...)` |
| `sql.connect.sql_formatter.SQLStringFormatter` | Class | `(session: SparkSession)` |
| `sql.connect.sql_formatter.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.streaming.query.CapturedStreamingQueryException` | Class | `(...)` |
| `sql.connect.streaming.query.PySparkStreamingQuery` | Class | `(jsq: JavaObject)` |
| `sql.connect.streaming.query.PySparkStreamingQueryManager` | Class | `(jsqm: JavaObject)` |
| `sql.connect.streaming.query.PySparkValueError` | Class | `(...)` |
| `sql.connect.streaming.query.QueryIdleEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, timestamp: str)` |
| `sql.connect.streaming.query.QueryProgressEvent` | Class | `(progress: StreamingQueryProgress)` |
| `sql.connect.streaming.query.QueryStartedEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, name: Optional[str], timestamp: str, jobTags: Set[str])` |
| `sql.connect.streaming.query.QueryTerminatedEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, exception: Optional[str], errorClassOnException: Optional[str])` |
| `sql.connect.streaming.query.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.streaming.query.StreamingQuery` | Class | `(session: SparkSession, queryId: str, runId: str, name: Optional[str])` |
| `sql.connect.streaming.query.StreamingQueryException` | Class | `(...)` |
| `sql.connect.streaming.query.StreamingQueryListener` | Class | `(...)` |
| `sql.connect.streaming.query.StreamingQueryListenerBus` | Class | `(sqm: StreamingQueryManager)` |
| `sql.connect.streaming.query.StreamingQueryManager` | Class | `(session: SparkSession)` |
| `sql.connect.streaming.query.StreamingQueryProgress` | Class | `(id: uuid.UUID, runId: uuid.UUID, name: Optional[str], timestamp: str, batchId: int, batchDuration: int, durationMs: Dict[str, int], eventTime: Dict[str, str], stateOperators: List[StateOperatorProgress], sources: List[SourceProgress], sink: SinkProgress, numInputRows: int, inputRowsPerSecond: float, processedRowsPerSecond: float, observedMetrics: Dict[str, Row], jprogress: Optional[JavaObject], jdict: Optional[Dict[str, Any]])` |
| `sql.connect.streaming.query.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.streaming.query.pb2` | Object | `` |
| `sql.connect.streaming.query.proto` | Object | `` |
| `sql.connect.streaming.readwriter.CloudPickleSerializer` | Class | `(...)` |
| `sql.connect.streaming.readwriter.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.streaming.readwriter.DataSource` | Class | `(format: Optional[str], schema: Optional[str], options: Optional[Mapping[str, str]], paths: Optional[List[str]], predicates: Optional[List[str]], is_streaming: Optional[bool])` |
| `sql.connect.streaming.readwriter.DataStreamReader` | Class | `(client: SparkSession)` |
| `sql.connect.streaming.readwriter.DataStreamWriter` | Class | `(plan: LogicalPlan, session: SparkSession)` |
| `sql.connect.streaming.readwriter.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `sql.connect.streaming.readwriter.OptionUtils` | Class | `(...)` |
| `sql.connect.streaming.readwriter.OptionalPrimitiveType` | Object | `` |
| `sql.connect.streaming.readwriter.PySparkDataStreamReader` | Class | `(spark: SparkSession)` |
| `sql.connect.streaming.readwriter.PySparkDataStreamWriter` | Class | `(df: DataFrame)` |
| `sql.connect.streaming.readwriter.PySparkPicklingError` | Class | `(...)` |
| `sql.connect.streaming.readwriter.PySparkTypeError` | Class | `(...)` |
| `sql.connect.streaming.readwriter.PySparkValueError` | Class | `(...)` |
| `sql.connect.streaming.readwriter.QueryStartedEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, name: Optional[str], timestamp: str, jobTags: Set[str])` |
| `sql.connect.streaming.readwriter.Read` | Class | `(table_name: str, options: Optional[Dict[str, str]], is_streaming: Optional[bool])` |
| `sql.connect.streaming.readwriter.Row` | Class | `(...)` |
| `sql.connect.streaming.readwriter.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.streaming.readwriter.StreamingQuery` | Class | `(session: SparkSession, queryId: str, runId: str, name: Optional[str])` |
| `sql.connect.streaming.readwriter.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.streaming.readwriter.SupportsProcess` | Class | `(...)` |
| `sql.connect.streaming.readwriter.WriteStreamOperation` | Class | `(child: LogicalPlan)` |
| `sql.connect.streaming.readwriter.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.streaming.readwriter.get_python_ver` | Function | `() -> str` |
| `sql.connect.streaming.readwriter.pb2` | Object | `` |
| `sql.connect.streaming.readwriter.to_str` | Function | `(value: Any) -> Optional[str]` |
| `sql.connect.streaming.worker.foreach_batch_worker.CPickleSerializer` | Object | `` |
| `sql.connect.streaming.worker.foreach_batch_worker.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.streaming.worker.foreach_batch_worker.UTF8Deserializer` | Class | `(use_unicode)` |
| `sql.connect.streaming.worker.foreach_batch_worker.auth_secret` | Object | `` |
| `sql.connect.streaming.worker.foreach_batch_worker.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.connect.streaming.worker.foreach_batch_worker.conn_info` | Object | `` |
| `sql.connect.streaming.worker.foreach_batch_worker.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.connect.streaming.worker.foreach_batch_worker.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.connect.streaming.worker.foreach_batch_worker.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.connect.streaming.worker.foreach_batch_worker.pickle_ser` | Object | `` |
| `sql.connect.streaming.worker.foreach_batch_worker.read_long` | Function | `(stream)` |
| `sql.connect.streaming.worker.foreach_batch_worker.spark` | Object | `` |
| `sql.connect.streaming.worker.foreach_batch_worker.utf8_deserializer` | Object | `` |
| `sql.connect.streaming.worker.foreach_batch_worker.worker` | Object | `` |
| `sql.connect.streaming.worker.foreach_batch_worker.write_int` | Function | `(value, stream)` |
| `sql.connect.streaming.worker.listener_worker.CPickleSerializer` | Object | `` |
| `sql.connect.streaming.worker.listener_worker.QueryIdleEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, timestamp: str)` |
| `sql.connect.streaming.worker.listener_worker.QueryProgressEvent` | Class | `(progress: StreamingQueryProgress)` |
| `sql.connect.streaming.worker.listener_worker.QueryStartedEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, name: Optional[str], timestamp: str, jobTags: Set[str])` |
| `sql.connect.streaming.worker.listener_worker.QueryTerminatedEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, exception: Optional[str], errorClassOnException: Optional[str])` |
| `sql.connect.streaming.worker.listener_worker.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.streaming.worker.listener_worker.UTF8Deserializer` | Class | `(use_unicode)` |
| `sql.connect.streaming.worker.listener_worker.auth_secret` | Object | `` |
| `sql.connect.streaming.worker.listener_worker.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.connect.streaming.worker.listener_worker.conn_info` | Object | `` |
| `sql.connect.streaming.worker.listener_worker.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.connect.streaming.worker.listener_worker.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.connect.streaming.worker.listener_worker.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.connect.streaming.worker.listener_worker.pickle_ser` | Object | `` |
| `sql.connect.streaming.worker.listener_worker.read_int` | Function | `(stream)` |
| `sql.connect.streaming.worker.listener_worker.spark` | Object | `` |
| `sql.connect.streaming.worker.listener_worker.utf8_deserializer` | Object | `` |
| `sql.connect.streaming.worker.listener_worker.worker` | Object | `` |
| `sql.connect.streaming.worker.listener_worker.write_int` | Function | `(value, stream)` |
| `sql.connect.table_arg.Column` | Class | `(...)` |
| `sql.connect.table_arg.ColumnOrName` | Object | `` |
| `sql.connect.table_arg.Expression` | Class | `()` |
| `sql.connect.table_arg.F` | Object | `` |
| `sql.connect.table_arg.IllegalArgumentException` | Class | `(...)` |
| `sql.connect.table_arg.ParentTableArg` | Class | `(...)` |
| `sql.connect.table_arg.SortOrder` | Class | `(child: Expression, ascending: bool, nullsFirst: bool)` |
| `sql.connect.table_arg.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.connect.table_arg.SubqueryExpression` | Class | `(plan: LogicalPlan, subquery_type: str, partition_spec: Optional[Sequence[Expression]], order_spec: Optional[Sequence[SortOrder]], with_single_partition: Optional[bool], in_subquery_values: Optional[Sequence[Expression]])` |
| `sql.connect.table_arg.TableArg` | Class | `(subquery_expr: SubqueryExpression)` |
| `sql.connect.table_arg.proto` | Object | `` |
| `sql.connect.tvf.Column` | Class | `(expr: Expression)` |
| `sql.connect.tvf.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.tvf.PySparkTableValuedFunction` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.tvf.PySparkValueError` | Class | `(...)` |
| `sql.connect.tvf.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.tvf.TableValuedFunction` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.types.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.connect.types.BinaryType` | Class | `(...)` |
| `sql.connect.types.BooleanType` | Class | `(...)` |
| `sql.connect.types.ByteType` | Class | `(...)` |
| `sql.connect.types.CalendarIntervalType` | Class | `(...)` |
| `sql.connect.types.CharType` | Class | `(length: int)` |
| `sql.connect.types.DataType` | Class | `(...)` |
| `sql.connect.types.DateType` | Class | `(...)` |
| `sql.connect.types.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `sql.connect.types.DecimalType` | Class | `(precision: int, scale: int)` |
| `sql.connect.types.DoubleType` | Class | `(...)` |
| `sql.connect.types.FloatType` | Class | `(...)` |
| `sql.connect.types.GeographyType` | Class | `(srid: int \| str)` |
| `sql.connect.types.GeometryType` | Class | `(srid: int \| str)` |
| `sql.connect.types.IntegerType` | Class | `(...)` |
| `sql.connect.types.LongType` | Class | `(...)` |
| `sql.connect.types.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `sql.connect.types.NullType` | Class | `(...)` |
| `sql.connect.types.NumericType` | Class | `(...)` |
| `sql.connect.types.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.connect.types.PySparkValueError` | Class | `(...)` |
| `sql.connect.types.ShortType` | Class | `(...)` |
| `sql.connect.types.StringType` | Class | `(collation: str)` |
| `sql.connect.types.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `sql.connect.types.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.types.TimeType` | Class | `(precision: int)` |
| `sql.connect.types.TimestampNTZType` | Class | `(...)` |
| `sql.connect.types.TimestampType` | Class | `(...)` |
| `sql.connect.types.UnparsedDataType` | Class | `(data_type_string: str)` |
| `sql.connect.types.UserDefinedType` | Class | `(...)` |
| `sql.connect.types.VarcharType` | Class | `(length: int)` |
| `sql.connect.types.VariantType` | Class | `(...)` |
| `sql.connect.types.YearMonthIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `sql.connect.types.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.types.parse_attr_name` | Function | `(name: str) -> Optional[List[str]]` |
| `sql.connect.types.pb2` | Object | `` |
| `sql.connect.types.proto_schema_to_pyspark_data_type` | Function | `(schema: pb2.DataType) -> DataType` |
| `sql.connect.types.pyspark_types_to_proto_types` | Function | `(data_type: DataType) -> pb2.DataType` |
| `sql.connect.types.verify_col_name` | Function | `(name: str, schema: StructType) -> bool` |
| `sql.connect.types.verify_numeric_col_name` | Function | `(name: str, schema: StructType) -> bool` |
| `sql.connect.udf.Column` | Class | `(expr: Expression)` |
| `sql.connect.udf.ColumnOrName` | Object | `` |
| `sql.connect.udf.ColumnReference` | Class | `(unparsed_identifier: str, plan_id: Optional[int], is_metadata_column: bool)` |
| `sql.connect.udf.CommonInlineUserDefinedFunction` | Class | `(function_name: str, function: Union[PythonUDF, JavaUDF], deterministic: bool, arguments: Optional[Sequence[Expression]])` |
| `sql.connect.udf.DataType` | Class | `(...)` |
| `sql.connect.udf.DataTypeOrString` | Object | `` |
| `sql.connect.udf.Expression` | Class | `()` |
| `sql.connect.udf.NamedArgumentExpression` | Class | `(key: str, value: Expression)` |
| `sql.connect.udf.PySparkRuntimeError` | Class | `(...)` |
| `sql.connect.udf.PySparkTypeError` | Class | `(...)` |
| `sql.connect.udf.PySparkUDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.udf.PySparkUserDefinedFunction` | Class | `(func: Callable[..., Any], returnType: DataTypeOrString, name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.connect.udf.PythonEvalType` | Class | `(...)` |
| `sql.connect.udf.PythonUDF` | Class | `(output_type: Union[DataType, str], eval_type: int, func: Callable[..., Any], python_ver: str)` |
| `sql.connect.udf.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.udf.StringType` | Class | `(collation: str)` |
| `sql.connect.udf.UDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.udf.UserDefinedFunction` | Class | `(func: Callable[..., Any], returnType: DataTypeOrString, name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.connect.udf.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.connect.udf.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.udf.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.connect.udf.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.connect.udtf.AnalyzeArgument` | Class | `(dataType: DataType, value: Optional[Any], isTable: bool, isConstantExpression: bool)` |
| `sql.connect.udtf.AnalyzeResult` | Class | `(schema: StructType, withSinglePartition: bool, partitionBy: Sequence[PartitioningColumn], orderBy: Sequence[OrderingColumn], select: Sequence[SelectedColumn])` |
| `sql.connect.udtf.Column` | Class | `(expr: Expression)` |
| `sql.connect.udtf.ColumnOrName` | Object | `` |
| `sql.connect.udtf.ColumnReference` | Class | `(unparsed_identifier: str, plan_id: Optional[int], is_metadata_column: bool)` |
| `sql.connect.udtf.CommonInlineUserDefinedTableFunction` | Class | `(function_name: str, function: PythonUDTF, deterministic: bool, arguments: Sequence[Expression])` |
| `sql.connect.udtf.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `sql.connect.udtf.DataType` | Class | `(...)` |
| `sql.connect.udtf.Expression` | Class | `()` |
| `sql.connect.udtf.NamedArgumentExpression` | Class | `(key: str, value: Expression)` |
| `sql.connect.udtf.PySparkAttributeError` | Class | `(...)` |
| `sql.connect.udtf.PySparkRuntimeError` | Class | `(...)` |
| `sql.connect.udtf.PySparkTypeError` | Class | `(...)` |
| `sql.connect.udtf.PySparkUDTFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.udtf.PythonEvalType` | Class | `(...)` |
| `sql.connect.udtf.PythonUDTF` | Class | `(func: Type, return_type: Optional[Union[DataType, str]], eval_type: int, python_ver: str)` |
| `sql.connect.udtf.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `sql.connect.udtf.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.connect.udtf.TableArg` | Class | `(subquery_expr: SubqueryExpression)` |
| `sql.connect.udtf.UDTFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.connect.udtf.UnparsedDataType` | Class | `(data_type_string: str)` |
| `sql.connect.udtf.UserDefinedTableFunction` | Class | `(func: Type, returnType: Optional[Union[StructType, str]], name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.connect.udtf.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.udtf.get_python_ver` | Function | `() -> str` |
| `sql.connect.udtf.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.connect.udtf.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.connect.utils.LooseVersion` | Class | `(vstring: Optional[str])` |
| `sql.connect.utils.PySparkImportError` | Class | `(...)` |
| `sql.connect.utils.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.connect.utils.get_python_ver` | Function | `() -> str` |
| `sql.connect.utils.require_minimum_googleapis_common_protos_version` | Function | `() -> None` |
| `sql.connect.utils.require_minimum_grpc_version` | Function | `() -> None` |
| `sql.connect.utils.require_minimum_grpcio_status_version` | Function | `() -> None` |
| `sql.connect.utils.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.connect.utils.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.connect.utils.require_minimum_zstandard_version` | Function | `() -> None` |
| `sql.connect.window.Column` | Class | `(...)` |
| `sql.connect.window.ColumnOrName` | Object | `` |
| `sql.connect.window.Expression` | Class | `()` |
| `sql.connect.window.F` | Object | `` |
| `sql.connect.window.ParentWindow` | Class | `(...)` |
| `sql.connect.window.ParentWindowSpec` | Class | `(...)` |
| `sql.connect.window.SortOrder` | Class | `(child: Expression, ascending: bool, nullsFirst: bool)` |
| `sql.connect.window.Window` | Class | `(...)` |
| `sql.connect.window.WindowFrame` | Class | `(isRowFrame: bool, start: int, end: int)` |
| `sql.connect.window.WindowSpec` | Class | `(partitionSpec: Sequence[Expression], orderSpec: Sequence[SortOrder], frame: Optional[WindowFrame])` |
| `sql.connect.window.check_dependencies` | Function | `(mod_name: str) -> None` |
| `sql.context.AtomicType` | Class | `(...)` |
| `sql.context.AtomicValue` | Object | `` |
| `sql.context.DataFrame` | Class | `(...)` |
| `sql.context.DataFrameReader` | Class | `(spark: SparkSession)` |
| `sql.context.DataStreamReader` | Class | `(spark: SparkSession)` |
| `sql.context.DataType` | Class | `(...)` |
| `sql.context.HiveContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jhiveContext: Optional[JavaObject])` |
| `sql.context.PandasDataFrameLike` | Object | `` |
| `sql.context.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `sql.context.RowLike` | Object | `` |
| `sql.context.SQLContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jsqlContext: Optional[JavaObject])` |
| `sql.context.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.context.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.context.StreamingQueryManager` | Class | `(jsqm: JavaObject)` |
| `sql.context.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.context.UDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.context.UDTFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.context.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.context.install_exception_handler` | Function | `() -> None` |
| `sql.conversion.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.conversion.ArrowTableToRowsConversion` | Class | `(...)` |
| `sql.conversion.BinaryType` | Class | `(...)` |
| `sql.conversion.DataType` | Class | `(...)` |
| `sql.conversion.DecimalType` | Class | `(precision: int, scale: int)` |
| `sql.conversion.Geography` | Class | `(wkb: bytes, srid: int)` |
| `sql.conversion.GeographyType` | Class | `(srid: int \| str)` |
| `sql.conversion.Geometry` | Class | `(wkb: bytes, srid: int)` |
| `sql.conversion.GeometryType` | Class | `(srid: int \| str)` |
| `sql.conversion.LocalDataToArrowConversion` | Class | `(...)` |
| `sql.conversion.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `sql.conversion.NullType` | Class | `(...)` |
| `sql.conversion.PySparkValueError` | Class | `(...)` |
| `sql.conversion.Row` | Class | `(...)` |
| `sql.conversion.StringType` | Class | `(collation: str)` |
| `sql.conversion.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `sql.conversion.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.conversion.TimestampNTZType` | Class | `(...)` |
| `sql.conversion.TimestampType` | Class | `(...)` |
| `sql.conversion.UserDefinedType` | Class | `(...)` |
| `sql.conversion.VariantType` | Class | `(...)` |
| `sql.conversion.VariantVal` | Class | `(value: bytes, metadata: bytes)` |
| `sql.conversion.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.conversion.to_arrow_schema` | Function | `(schema: StructType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.Schema` |
| `sql.dataframe.ArrowMapIterFunction` | Object | `` |
| `sql.dataframe.Column` | Class | `(...)` |
| `sql.dataframe.ColumnOrName` | Object | `` |
| `sql.dataframe.ColumnOrNameOrOrdinal` | Object | `` |
| `sql.dataframe.DataFrame` | Class | `(...)` |
| `sql.dataframe.DataFrameNaFunctions` | Class | `(df: DataFrame)` |
| `sql.dataframe.DataFrameStatFunctions` | Class | `(df: DataFrame)` |
| `sql.dataframe.DataFrameWriter` | Class | `(df: DataFrame)` |
| `sql.dataframe.DataFrameWriterV2` | Class | `(df: DataFrame, table: str)` |
| `sql.dataframe.DataStreamWriter` | Class | `(df: DataFrame)` |
| `sql.dataframe.ExecutionInfo` | Class | `(metrics: Optional[list[PlanMetrics]], obs: Optional[Sequence[ObservedMetrics]])` |
| `sql.dataframe.GroupedData` | Class | `(jgd: JavaObject, df: DataFrame)` |
| `sql.dataframe.LiteralType` | Object | `` |
| `sql.dataframe.MergeIntoWriter` | Class | `(df: DataFrame, table: str, condition: Column)` |
| `sql.dataframe.Observation` | Class | `(name: Optional[str])` |
| `sql.dataframe.OptionalPrimitiveType` | Object | `` |
| `sql.dataframe.PandasDataFrameLike` | Object | `` |
| `sql.dataframe.PandasMapIterFunction` | Object | `` |
| `sql.dataframe.PandasOnSparkDataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `sql.dataframe.PrimitiveType` | Object | `` |
| `sql.dataframe.PySparkPlotAccessor` | Class | `(data: DataFrame)` |
| `sql.dataframe.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `sql.dataframe.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `sql.dataframe.Row` | Class | `(...)` |
| `sql.dataframe.SQLContext` | Class | `(sparkContext: SparkContext, sparkSession: Optional[SparkSession], jsqlContext: Optional[JavaObject])` |
| `sql.dataframe.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.dataframe.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.dataframe.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `sql.dataframe.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.dataframe.TableArg` | Class | `(...)` |
| `sql.dataframe.dispatch_df_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.dataframe.is_remote_only` | Function | `() -> bool` |
| `sql.datasource.CaseInsensitiveDict` | Class | `(args: Any, kwargs: Any)` |
| `sql.datasource.ColumnPath` | Object | `` |
| `sql.datasource.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.datasource.DataSourceArrowWriter` | Class | `(...)` |
| `sql.datasource.DataSourceReader` | Class | `(...)` |
| `sql.datasource.DataSourceRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.datasource.DataSourceStreamArrowWriter` | Class | `(...)` |
| `sql.datasource.DataSourceStreamReader` | Class | `(...)` |
| `sql.datasource.DataSourceStreamWriter` | Class | `(...)` |
| `sql.datasource.DataSourceWriter` | Class | `(...)` |
| `sql.datasource.EqualNullSafe` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.datasource.EqualTo` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.datasource.Filter` | Class | `()` |
| `sql.datasource.GreaterThan` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.datasource.GreaterThanOrEqual` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.datasource.In` | Class | `(attribute: ColumnPath, value: Tuple[Any, ...])` |
| `sql.datasource.InputPartition` | Class | `(value: Any)` |
| `sql.datasource.IsNotNull` | Class | `(attribute: ColumnPath)` |
| `sql.datasource.IsNull` | Class | `(attribute: ColumnPath)` |
| `sql.datasource.LessThan` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.datasource.LessThanOrEqual` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.datasource.Not` | Class | `(child: Filter)` |
| `sql.datasource.PySparkNotImplementedError` | Class | `(...)` |
| `sql.datasource.Row` | Class | `(...)` |
| `sql.datasource.SimpleDataSourceStreamReader` | Class | `(...)` |
| `sql.datasource.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.datasource.StringContains` | Class | `(attribute: ColumnPath, value: str)` |
| `sql.datasource.StringEndsWith` | Class | `(attribute: ColumnPath, value: str)` |
| `sql.datasource.StringStartsWith` | Class | `(attribute: ColumnPath, value: str)` |
| `sql.datasource.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.datasource.WriterCommitMessage` | Class | `(...)` |
| `sql.datasource_internal.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.datasource_internal.DataSourceStreamReader` | Class | `(...)` |
| `sql.datasource_internal.InputPartition` | Class | `(value: Any)` |
| `sql.datasource_internal.PrefetchedCacheEntry` | Class | `(start: dict, end: dict, iterator: Iterator[Tuple])` |
| `sql.datasource_internal.PySparkNotImplementedError` | Class | `(...)` |
| `sql.datasource_internal.SimpleDataSourceStreamReader` | Class | `(...)` |
| `sql.datasource_internal.SimpleInputPartition` | Class | `(start: dict, end: dict)` |
| `sql.datasource_internal.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.functions.AnalyzeArgument` | Class | `(dataType: DataType, value: Optional[Any], isTable: bool, isConstantExpression: bool)` |
| `sql.functions.AnalyzeResult` | Class | `(schema: StructType, withSinglePartition: bool, partitionBy: Sequence[PartitioningColumn], orderBy: Sequence[OrderingColumn], select: Sequence[SelectedColumn])` |
| `sql.functions.Any` | Object | `` |
| `sql.functions.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.functions.ArrowUDFType` | Class | `(...)` |
| `sql.functions.ByteType` | Class | `(...)` |
| `sql.functions.Callable` | Object | `` |
| `sql.functions.Column` | Class | `(...)` |
| `sql.functions.DataType` | Class | `(...)` |
| `sql.functions.Iterable` | Object | `` |
| `sql.functions.Mapping` | Object | `` |
| `sql.functions.NumericType` | Class | `(...)` |
| `sql.functions.Optional` | Object | `` |
| `sql.functions.OrderingColumn` | Class | `(name: str, ascending: bool, overrideNullsFirst: Optional[bool])` |
| `sql.functions.PandasUDFType` | Class | `(...)` |
| `sql.functions.PartitioningColumn` | Class | `(name: str)` |
| `sql.functions.PySparkTypeError` | Class | `(...)` |
| `sql.functions.PySparkValueError` | Class | `(...)` |
| `sql.functions.SelectedColumn` | Class | `(name: str, alias: str)` |
| `sql.functions.Sequence` | Object | `` |
| `sql.functions.SkipRestOfInputTableException` | Class | `(...)` |
| `sql.functions.StringType` | Class | `(collation: str)` |
| `sql.functions.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.functions.TYPE_CHECKING` | Object | `` |
| `sql.functions.Tuple` | Object | `` |
| `sql.functions.Type` | Object | `` |
| `sql.functions.Union` | Object | `` |
| `sql.functions.UserDefinedFunction` | Class | `(func: Callable[..., Any], returnType: DataTypeOrString, name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.functions.UserDefinedTableFunction` | Class | `(func: Type, returnType: Optional[Union[StructType, str]], name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.functions.ValuesView` | Object | `` |
| `sql.functions.abs` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.acos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.acosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.add_months` | Function | `(start: ColumnOrName, months: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.functions.aes_encrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], iv: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.functions.aggregate` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.functions.any_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.approxCountDistinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.functions.approx_count_distinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.functions.approx_percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.functions.array` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.array_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.array_append` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.array_compact` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.array_contains` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.array_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.array_except` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.array_insert` | Function | `(arr: ColumnOrName, pos: Union[ColumnOrName, int], value: Any) -> Column` |
| `sql.functions.array_intersect` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.array_join` | Function | `(col: ColumnOrName, delimiter: str, null_replacement: Optional[str]) -> Column` |
| `sql.functions.array_max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.array_min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.array_position` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.array_prepend` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.array_remove` | Function | `(col: ColumnOrName, element: Any) -> Column` |
| `sql.functions.array_repeat` | Function | `(col: ColumnOrName, count: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.array_size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.array_sort` | Function | `(col: ColumnOrName, comparator: Optional[Callable[[Column, Column], Column]]) -> Column` |
| `sql.functions.array_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.arrays_overlap` | Function | `(a1: ColumnOrName, a2: ColumnOrName) -> Column` |
| `sql.functions.arrays_zip` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.arrow_udf` | Function | `(f: ArrowScalarToScalarFunction \| DataTypeOrString \| ArrowScalarIterFunction \| (AtomicDataTypeOrString \| ArrayType), returnType: DataTypeOrString \| ArrowScalarUDFType \| (AtomicDataTypeOrString \| ArrayType) \| ArrowScalarIterUDFType, functionType: ArrowScalarUDFType \| ArrowScalarIterUDFType) -> UserDefinedFunctionLike \| Callable[[ArrowScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[ArrowScalarIterFunction], UserDefinedFunctionLike]` |
| `sql.functions.arrow_udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.functions.asc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.asc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.asc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.ascii` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.asin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.asinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.assert_true` | Function | `(col: ColumnOrName, errMsg: Optional[Union[Column, str]]) -> Column` |
| `sql.functions.atan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.atan2` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.atanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.base64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bit_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bit_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bit_get` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.functions.bit_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bit_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bit_xor` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitmap_and_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitmap_bit_position` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitmap_bucket_number` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitmap_construct_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitmap_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitmap_or_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitwiseNOT` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bitwise_not` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bool_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.bool_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.broadcast` | Function | `(df: DataFrame) -> DataFrame` |
| `sql.functions.bround` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.btrim` | Function | `(str: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.bucket` | Function | `(numBuckets: Union[Column, int], col: ColumnOrName) -> Column` |
| `sql.functions.builtin.AnalyzeArgument` | Class | `(dataType: DataType, value: Optional[Any], isTable: bool, isConstantExpression: bool)` |
| `sql.functions.builtin.AnalyzeResult` | Class | `(schema: StructType, withSinglePartition: bool, partitionBy: Sequence[PartitioningColumn], orderBy: Sequence[OrderingColumn], select: Sequence[SelectedColumn])` |
| `sql.functions.builtin.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.functions.builtin.ArrowUDFType` | Class | `(...)` |
| `sql.functions.builtin.ByteType` | Class | `(...)` |
| `sql.functions.builtin.Column` | Class | `(...)` |
| `sql.functions.builtin.ColumnOrName` | Object | `` |
| `sql.functions.builtin.DataFrame` | Class | `(...)` |
| `sql.functions.builtin.DataType` | Class | `(...)` |
| `sql.functions.builtin.DataTypeOrString` | Object | `` |
| `sql.functions.builtin.NumericType` | Class | `(...)` |
| `sql.functions.builtin.OrderingColumn` | Class | `(name: str, ascending: bool, overrideNullsFirst: Optional[bool])` |
| `sql.functions.builtin.PandasUDFType` | Class | `(...)` |
| `sql.functions.builtin.PartitioningColumn` | Class | `(name: str)` |
| `sql.functions.builtin.PySparkTypeError` | Class | `(...)` |
| `sql.functions.builtin.PySparkValueError` | Class | `(...)` |
| `sql.functions.builtin.SelectedColumn` | Class | `(name: str, alias: str)` |
| `sql.functions.builtin.SkipRestOfInputTableException` | Class | `(...)` |
| `sql.functions.builtin.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.functions.builtin.StringType` | Class | `(collation: str)` |
| `sql.functions.builtin.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.functions.builtin.UserDefinedFunction` | Class | `(func: Callable[..., Any], returnType: DataTypeOrString, name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.functions.builtin.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.functions.builtin.UserDefinedTableFunction` | Class | `(func: Type, returnType: Optional[Union[StructType, str]], name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.functions.builtin.abs` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.acos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.acosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.add_months` | Function | `(start: ColumnOrName, months: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.aes_encrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], iv: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.aggregate` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.functions.builtin.any_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.builtin.approxCountDistinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.functions.builtin.approx_count_distinct` | Function | `(col: ColumnOrName, rsd: Optional[float]) -> Column` |
| `sql.functions.builtin.approx_percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.functions.builtin.array` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.builtin.array_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_append` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.builtin.array_compact` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_contains` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.builtin.array_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_except` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_insert` | Function | `(arr: ColumnOrName, pos: Union[ColumnOrName, int], value: Any) -> Column` |
| `sql.functions.builtin.array_intersect` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_join` | Function | `(col: ColumnOrName, delimiter: str, null_replacement: Optional[str]) -> Column` |
| `sql.functions.builtin.array_max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_position` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.builtin.array_prepend` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.builtin.array_remove` | Function | `(col: ColumnOrName, element: Any) -> Column` |
| `sql.functions.builtin.array_repeat` | Function | `(col: ColumnOrName, count: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.array_size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.array_sort` | Function | `(col: ColumnOrName, comparator: Optional[Callable[[Column, Column], Column]]) -> Column` |
| `sql.functions.builtin.array_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.arrays_overlap` | Function | `(a1: ColumnOrName, a2: ColumnOrName) -> Column` |
| `sql.functions.builtin.arrays_zip` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.arrow_udf` | Function | `(f: ArrowScalarToScalarFunction \| DataTypeOrString \| ArrowScalarIterFunction \| (AtomicDataTypeOrString \| ArrayType), returnType: DataTypeOrString \| ArrowScalarUDFType \| (AtomicDataTypeOrString \| ArrayType) \| ArrowScalarIterUDFType, functionType: ArrowScalarUDFType \| ArrowScalarIterUDFType) -> UserDefinedFunctionLike \| Callable[[ArrowScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[ArrowScalarIterFunction], UserDefinedFunctionLike]` |
| `sql.functions.builtin.arrow_udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.functions.builtin.asc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.asc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.asc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.ascii` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.asin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.asinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.assert_true` | Function | `(col: ColumnOrName, errMsg: Optional[Union[Column, str]]) -> Column` |
| `sql.functions.builtin.atan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.atan2` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.builtin.atanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.base64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bit_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bit_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bit_get` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.functions.builtin.bit_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bit_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bit_xor` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitmap_and_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitmap_bit_position` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitmap_bucket_number` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitmap_construct_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitmap_count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitmap_or_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitwiseNOT` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bitwise_not` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bool_and` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.bool_or` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.broadcast` | Function | `(df: DataFrame) -> DataFrame` |
| `sql.functions.builtin.bround` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.btrim` | Function | `(str: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.bucket` | Function | `(numBuckets: Union[Column, int], col: ColumnOrName) -> Column` |
| `sql.functions.builtin.call_function` | Function | `(funcName: str, cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.call_udf` | Function | `(udfName: str, cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.cardinality` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.cbrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.ceil` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.ceiling` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.char` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.char_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.character_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.chr` | Function | `(n: ColumnOrName) -> Column` |
| `sql.functions.builtin.coalesce` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.col` | Function | `(col: str) -> Column` |
| `sql.functions.builtin.collate` | Function | `(col: ColumnOrName, collation: str) -> Column` |
| `sql.functions.builtin.collation` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.collect_list` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.collect_set` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.column` | Object | `` |
| `sql.functions.builtin.concat` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.concat_ws` | Function | `(sep: str, cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.contains` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.conv` | Function | `(col: ColumnOrName, fromBase: int, toBase: int) -> Column` |
| `sql.functions.builtin.convert_timezone` | Function | `(sourceTz: Optional[Column], targetTz: Column, sourceTs: ColumnOrName) -> Column` |
| `sql.functions.builtin.corr` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.cos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.cosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.cot` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.countDistinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.count_distinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.count_if` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.count_min_sketch` | Function | `(col: ColumnOrName, eps: Union[Column, float], confidence: Union[Column, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.covar_pop` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.covar_samp` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.crc32` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.create_map` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.builtin.csc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.cume_dist` | Function | `() -> Column` |
| `sql.functions.builtin.curdate` | Function | `() -> Column` |
| `sql.functions.builtin.current_catalog` | Function | `() -> Column` |
| `sql.functions.builtin.current_database` | Function | `() -> Column` |
| `sql.functions.builtin.current_date` | Function | `() -> Column` |
| `sql.functions.builtin.current_schema` | Function | `() -> Column` |
| `sql.functions.builtin.current_time` | Function | `(precision: Optional[int]) -> Column` |
| `sql.functions.builtin.current_timestamp` | Function | `() -> Column` |
| `sql.functions.builtin.current_timezone` | Function | `() -> Column` |
| `sql.functions.builtin.current_user` | Function | `() -> Column` |
| `sql.functions.builtin.date_add` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.date_diff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.functions.builtin.date_format` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.functions.builtin.date_from_unix_date` | Function | `(days: ColumnOrName) -> Column` |
| `sql.functions.builtin.date_part` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.functions.builtin.date_sub` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.date_trunc` | Function | `(format: str, timestamp: ColumnOrName) -> Column` |
| `sql.functions.builtin.dateadd` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.datediff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.functions.builtin.datepart` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.functions.builtin.day` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.dayname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.dayofmonth` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.dayofweek` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.dayofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.days` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.decode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.functions.builtin.degrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.dense_rank` | Function | `() -> Column` |
| `sql.functions.builtin.desc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.desc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.desc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.e` | Function | `() -> Column` |
| `sql.functions.builtin.element_at` | Function | `(col: ColumnOrName, extraction: Any) -> Column` |
| `sql.functions.builtin.elt` | Function | `(inputs: ColumnOrName) -> Column` |
| `sql.functions.builtin.encode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.functions.builtin.endswith` | Function | `(str: ColumnOrName, suffix: ColumnOrName) -> Column` |
| `sql.functions.builtin.equal_null` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.every` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.exists` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.functions.builtin.exp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.explode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.explode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.expm1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.expr` | Function | `(str: str) -> Column` |
| `sql.functions.builtin.extract` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.functions.builtin.factorial` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.filter` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.functions.builtin.find_in_set` | Function | `(str: ColumnOrName, str_array: ColumnOrName) -> Column` |
| `sql.functions.builtin.first` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.functions.builtin.first_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.builtin.flatten` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.floor` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.forall` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.functions.builtin.format_number` | Function | `(col: ColumnOrName, d: int) -> Column` |
| `sql.functions.builtin.format_string` | Function | `(format: str, cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.from_csv` | Function | `(col: ColumnOrName, schema: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.from_json` | Function | `(col: ColumnOrName, schema: Union[ArrayType, StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.from_unixtime` | Function | `(timestamp: ColumnOrName, format: str) -> Column` |
| `sql.functions.builtin.from_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.functions.builtin.from_xml` | Function | `(col: ColumnOrName, schema: Union[StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.get` | Function | `(col: ColumnOrName, index: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.get_json_object` | Function | `(col: ColumnOrName, path: str) -> Column` |
| `sql.functions.builtin.getbit` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.functions.builtin.greatest` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.grouping` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.grouping_id` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.hash` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.hex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.histogram_numeric` | Function | `(col: ColumnOrName, nBins: Column) -> Column` |
| `sql.functions.builtin.hll_sketch_agg` | Function | `(col: ColumnOrName, lgConfigK: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.hll_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.hll_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, allowDifferentLgConfigK: Optional[bool]) -> Column` |
| `sql.functions.builtin.hll_union_agg` | Function | `(col: ColumnOrName, allowDifferentLgConfigK: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.builtin.hour` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.hours` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.hypot` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.builtin.ifnull` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.ilike` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.functions.builtin.initcap` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.inline` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.inline_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.input_file_block_length` | Function | `() -> Column` |
| `sql.functions.builtin.input_file_block_start` | Function | `() -> Column` |
| `sql.functions.builtin.input_file_name` | Function | `() -> Column` |
| `sql.functions.builtin.instr` | Function | `(str: ColumnOrName, substr: Union[Column, str]) -> Column` |
| `sql.functions.builtin.is_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.is_variant_null` | Function | `(v: ColumnOrName) -> Column` |
| `sql.functions.builtin.isnan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.isnotnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.isnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.java_method` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.json_array_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.json_object_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.json_tuple` | Function | `(col: ColumnOrName, fields: str) -> Column` |
| `sql.functions.builtin.kll_sketch_agg_bigint` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.kll_sketch_agg_double` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.kll_sketch_agg_float` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.kll_sketch_get_n_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_n_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_n_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_quantile_bigint` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_quantile_double` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_quantile_float` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_rank_bigint` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_rank_double` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_get_rank_float` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_merge_bigint` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_merge_double` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_merge_float` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_to_string_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_to_string_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.kll_sketch_to_string_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.kurtosis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.lag` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.functions.builtin.last` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.functions.builtin.last_day` | Function | `(date: ColumnOrName) -> Column` |
| `sql.functions.builtin.last_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.builtin.lcase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.lead` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.functions.builtin.least` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.left` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.functions.builtin.length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.levenshtein` | Function | `(left: ColumnOrName, right: ColumnOrName, threshold: Optional[int]) -> Column` |
| `sql.functions.builtin.like` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.functions.builtin.listagg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.builtin.listagg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.builtin.lit` | Function | `(col: Any) -> Column` |
| `sql.functions.builtin.ln` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.localtimestamp` | Function | `() -> Column` |
| `sql.functions.builtin.locate` | Function | `(substr: str, str: ColumnOrName, pos: int) -> Column` |
| `sql.functions.builtin.log` | Function | `(arg1: Union[ColumnOrName, float], arg2: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.log10` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.log1p` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.log2` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.lower` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.lpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.functions.builtin.ltrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.make_date` | Function | `(year: ColumnOrName, month: ColumnOrName, day: ColumnOrName) -> Column` |
| `sql.functions.builtin.make_dt_interval` | Function | `(days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.make_time` | Function | `(hour: ColumnOrName, minute: ColumnOrName, second: ColumnOrName) -> Column` |
| `sql.functions.builtin.make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.make_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.make_ym_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.map_concat` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.builtin.map_contains_key` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.builtin.map_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.map_filter` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.functions.builtin.map_from_arrays` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.map_from_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.map_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.map_values` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.map_zip_with` | Function | `(col1: ColumnOrName, col2: ColumnOrName, f: Callable[[Column, Column, Column], Column]) -> Column` |
| `sql.functions.builtin.mask` | Function | `(col: ColumnOrName, upperChar: Optional[ColumnOrName], lowerChar: Optional[ColumnOrName], digitChar: Optional[ColumnOrName], otherChar: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.max_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.functions.builtin.md5` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.mean` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.median` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.min_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.functions.builtin.minute` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.mode` | Function | `(col: ColumnOrName, deterministic: bool) -> Column` |
| `sql.functions.builtin.monotonically_increasing_id` | Function | `() -> Column` |
| `sql.functions.builtin.month` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.monthname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.months` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.months_between` | Function | `(date1: ColumnOrName, date2: ColumnOrName, roundOff: bool) -> Column` |
| `sql.functions.builtin.named_struct` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.nanvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.negate` | Object | `` |
| `sql.functions.builtin.negative` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.next_day` | Function | `(date: ColumnOrName, dayOfWeek: str) -> Column` |
| `sql.functions.builtin.now` | Function | `() -> Column` |
| `sql.functions.builtin.nth_value` | Function | `(col: ColumnOrName, offset: int, ignoreNulls: Optional[bool]) -> Column` |
| `sql.functions.builtin.ntile` | Function | `(n: int) -> Column` |
| `sql.functions.builtin.nullif` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.nullifzero` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.nvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.nvl2` | Function | `(col1: ColumnOrName, col2: ColumnOrName, col3: ColumnOrName) -> Column` |
| `sql.functions.builtin.octet_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.overlay` | Function | `(src: ColumnOrName, replace: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `sql.functions.builtin.parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.percent_rank` | Function | `() -> Column` |
| `sql.functions.builtin.percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], frequency: Union[Column, int]) -> Column` |
| `sql.functions.builtin.percentile_approx` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.functions.builtin.pi` | Function | `() -> Column` |
| `sql.functions.builtin.pmod` | Function | `(dividend: Union[ColumnOrName, float], divisor: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.builtin.posexplode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.posexplode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.position` | Function | `(substr: ColumnOrName, str: ColumnOrName, start: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.positive` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.pow` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.builtin.power` | Object | `` |
| `sql.functions.builtin.printf` | Function | `(format: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.product` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.quarter` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.quote` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.radians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.raise_error` | Function | `(errMsg: Union[Column, str]) -> Column` |
| `sql.functions.builtin.rand` | Function | `(seed: Optional[int]) -> Column` |
| `sql.functions.builtin.randn` | Function | `(seed: Optional[int]) -> Column` |
| `sql.functions.builtin.random` | Object | `` |
| `sql.functions.builtin.randstr` | Function | `(length: Union[Column, int], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.rank` | Function | `() -> Column` |
| `sql.functions.builtin.reduce` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.functions.builtin.reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.regexp` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.builtin.regexp_count` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.builtin.regexp_extract` | Function | `(str: ColumnOrName, pattern: str, idx: int) -> Column` |
| `sql.functions.builtin.regexp_extract_all` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.regexp_instr` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.regexp_like` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.builtin.regexp_replace` | Function | `(string: ColumnOrName, pattern: Union[str, Column], replacement: Union[str, Column]) -> Column` |
| `sql.functions.builtin.regexp_substr` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_avgx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_avgy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_count` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_intercept` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_r2` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_slope` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_sxx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_sxy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.regr_syy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.builtin.repeat` | Function | `(col: ColumnOrName, n: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.replace` | Function | `(src: ColumnOrName, search: ColumnOrName, replace: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.reverse` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.right` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.functions.builtin.rint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.rlike` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.builtin.round` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.row_number` | Function | `() -> Column` |
| `sql.functions.builtin.rpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.functions.builtin.rtrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.schema_of_csv` | Function | `(csv: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.schema_of_json` | Function | `(json: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.schema_of_variant` | Function | `(v: ColumnOrName) -> Column` |
| `sql.functions.builtin.schema_of_variant_agg` | Function | `(v: ColumnOrName) -> Column` |
| `sql.functions.builtin.schema_of_xml` | Function | `(xml: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.sec` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.second` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sentences` | Function | `(string: ColumnOrName, language: Optional[ColumnOrName], country: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.sequence` | Function | `(start: ColumnOrName, stop: ColumnOrName, step: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.session_user` | Function | `() -> Column` |
| `sql.functions.builtin.session_window` | Function | `(timeColumn: ColumnOrName, gapDuration: Union[Column, str]) -> Column` |
| `sql.functions.builtin.sha` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sha1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sha2` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.builtin.shiftLeft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.builtin.shiftRight` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.builtin.shiftRightUnsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.builtin.shiftleft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.builtin.shiftright` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.builtin.shiftrightunsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.builtin.shuffle` | Function | `(col: ColumnOrName, seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.sign` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.signum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.skewness` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.slice` | Function | `(x: ColumnOrName, start: Union[ColumnOrName, int], length: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.some` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sort_array` | Function | `(col: ColumnOrName, asc: bool) -> Column` |
| `sql.functions.builtin.soundex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.spark_partition_id` | Function | `() -> Column` |
| `sql.functions.builtin.split` | Function | `(str: ColumnOrName, pattern: Union[Column, str], limit: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.split_part` | Function | `(src: ColumnOrName, delimiter: ColumnOrName, partNum: ColumnOrName) -> Column` |
| `sql.functions.builtin.sqrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.st_asbinary` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.functions.builtin.st_geogfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.functions.builtin.st_geomfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.functions.builtin.st_setsrid` | Function | `(geo: ColumnOrName, srid: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.st_srid` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.functions.builtin.stack` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.startswith` | Function | `(str: ColumnOrName, prefix: ColumnOrName) -> Column` |
| `sql.functions.builtin.std` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.stddev` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.stddev_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.stddev_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.str_to_map` | Function | `(text: ColumnOrName, pairDelim: Optional[ColumnOrName], keyValueDelim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.string_agg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.builtin.string_agg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.builtin.struct` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.builtin.substr` | Function | `(str: ColumnOrName, pos: ColumnOrName, len: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.substring` | Function | `(str: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.substring_index` | Function | `(str: ColumnOrName, delim: str, count: int) -> Column` |
| `sql.functions.builtin.sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sumDistinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.sum_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.tan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.tanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.theta_difference` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.theta_intersection` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.builtin.theta_intersection_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.theta_sketch_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.theta_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.theta_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.theta_union_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.builtin.time_diff` | Function | `(unit: ColumnOrName, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.functions.builtin.time_trunc` | Function | `(unit: ColumnOrName, time: ColumnOrName) -> Column` |
| `sql.functions.builtin.timestamp_add` | Function | `(unit: str, quantity: ColumnOrName, ts: ColumnOrName) -> Column` |
| `sql.functions.builtin.timestamp_diff` | Function | `(unit: str, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.functions.builtin.timestamp_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.timestamp_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.timestamp_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.toDegrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.toRadians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.to_char` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.builtin.to_csv` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.functions.builtin.to_json` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.builtin.to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.to_timestamp` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.functions.builtin.to_timestamp_ltz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.to_timestamp_ntz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.to_unix_timestamp` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.to_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.functions.builtin.to_varchar` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.builtin.to_variant_object` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.to_xml` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.builtin.transform` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.functions.builtin.transform_keys` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.functions.builtin.transform_values` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.functions.builtin.translate` | Function | `(srcCol: ColumnOrName, matching: str, replace: str) -> Column` |
| `sql.functions.builtin.trim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.trunc` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.functions.builtin.try_add` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_divide` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_element_at` | Function | `(col: ColumnOrName, extraction: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_mod` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_multiply` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_subtract` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.functions.builtin.try_to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_to_timestamp` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.builtin.try_url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.try_variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.functions.builtin.typeof` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.ucase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.udf` | Function | `(f: Optional[Union[Callable[..., Any], DataTypeOrString]], returnType: DataTypeOrString, useArrow: Optional[bool]) -> Union[UserDefinedFunctionLike, Callable[[Callable[..., Any]], UserDefinedFunctionLike]]` |
| `sql.functions.builtin.udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]], useArrow: Optional[bool]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.functions.builtin.unbase64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.unhex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.uniform` | Function | `(min: Union[Column, int, float], max: Union[Column, int, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.unix_date` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.unix_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.unix_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.unix_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.unix_timestamp` | Function | `(timestamp: Optional[ColumnOrName], format: str) -> Column` |
| `sql.functions.builtin.unwrap_udt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.upper` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.url_encode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.user` | Function | `() -> Column` |
| `sql.functions.builtin.uuid` | Function | `(seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.builtin.validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.builtin.var_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.var_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.variance` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.functions.builtin.version` | Function | `() -> Column` |
| `sql.functions.builtin.weekday` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.weekofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.when` | Function | `(condition: Column, value: Any) -> Column` |
| `sql.functions.builtin.width_bucket` | Function | `(v: ColumnOrName, min: ColumnOrName, max: ColumnOrName, numBucket: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.builtin.window` | Function | `(timeColumn: ColumnOrName, windowDuration: str, slideDuration: Optional[str], startTime: Optional[str]) -> Column` |
| `sql.functions.builtin.window_time` | Function | `(windowColumn: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_boolean` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_double` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_float` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_int` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_long` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_number` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_short` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xpath_string` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.builtin.xxhash64` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.builtin.year` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.years` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.zeroifnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.builtin.zip_with` | Function | `(left: ColumnOrName, right: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.functions.call_function` | Function | `(funcName: str, cols: ColumnOrName) -> Column` |
| `sql.functions.call_udf` | Function | `(udfName: str, cols: ColumnOrName) -> Column` |
| `sql.functions.cardinality` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.cast` | Object | `` |
| `sql.functions.cbrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.ceil` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.ceiling` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.char` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.char_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.character_length` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.chr` | Function | `(n: ColumnOrName) -> Column` |
| `sql.functions.coalesce` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.col` | Function | `(col: str) -> Column` |
| `sql.functions.collate` | Function | `(col: ColumnOrName, collation: str) -> Column` |
| `sql.functions.collation` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.collect_list` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.collect_set` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.column` | Object | `` |
| `sql.functions.concat` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.concat_ws` | Function | `(sep: str, cols: ColumnOrName) -> Column` |
| `sql.functions.contains` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.conv` | Function | `(col: ColumnOrName, fromBase: int, toBase: int) -> Column` |
| `sql.functions.convert_timezone` | Function | `(sourceTz: Optional[Column], targetTz: Column, sourceTs: ColumnOrName) -> Column` |
| `sql.functions.corr` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.cos` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.cosh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.cot` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.count` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.countDistinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.functions.count_distinct` | Function | `(col: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.functions.count_if` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.count_min_sketch` | Function | `(col: ColumnOrName, eps: Union[Column, float], confidence: Union[Column, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.covar_pop` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.covar_samp` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.crc32` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.create_map` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.csc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.cume_dist` | Function | `() -> Column` |
| `sql.functions.curdate` | Function | `() -> Column` |
| `sql.functions.current_catalog` | Function | `() -> Column` |
| `sql.functions.current_database` | Function | `() -> Column` |
| `sql.functions.current_date` | Function | `() -> Column` |
| `sql.functions.current_schema` | Function | `() -> Column` |
| `sql.functions.current_time` | Function | `(precision: Optional[int]) -> Column` |
| `sql.functions.current_timestamp` | Function | `() -> Column` |
| `sql.functions.current_timezone` | Function | `() -> Column` |
| `sql.functions.current_user` | Function | `() -> Column` |
| `sql.functions.date_add` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.date_diff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.functions.date_format` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.functions.date_from_unix_date` | Function | `(days: ColumnOrName) -> Column` |
| `sql.functions.date_part` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.functions.date_sub` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.date_trunc` | Function | `(format: str, timestamp: ColumnOrName) -> Column` |
| `sql.functions.dateadd` | Function | `(start: ColumnOrName, days: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.datediff` | Function | `(end: ColumnOrName, start: ColumnOrName) -> Column` |
| `sql.functions.datepart` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.functions.day` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.dayname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.dayofmonth` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.dayofweek` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.dayofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.days` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.decimal` | Object | `` |
| `sql.functions.decode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.functions.degrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.dense_rank` | Function | `() -> Column` |
| `sql.functions.desc` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.desc_nulls_first` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.desc_nulls_last` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.e` | Function | `() -> Column` |
| `sql.functions.element_at` | Function | `(col: ColumnOrName, extraction: Any) -> Column` |
| `sql.functions.elt` | Function | `(inputs: ColumnOrName) -> Column` |
| `sql.functions.encode` | Function | `(col: ColumnOrName, charset: str) -> Column` |
| `sql.functions.endswith` | Function | `(str: ColumnOrName, suffix: ColumnOrName) -> Column` |
| `sql.functions.equal_null` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.every` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.exists` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.functions.exp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.explode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.explode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.expm1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.expr` | Function | `(str: str) -> Column` |
| `sql.functions.extract` | Function | `(field: Column, source: ColumnOrName) -> Column` |
| `sql.functions.factorial` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.filter` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.functions.find_in_set` | Function | `(str: ColumnOrName, str_array: ColumnOrName) -> Column` |
| `sql.functions.first` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.functions.first_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.flatten` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.floor` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.forall` | Function | `(col: ColumnOrName, f: Callable[[Column], Column]) -> Column` |
| `sql.functions.format_number` | Function | `(col: ColumnOrName, d: int) -> Column` |
| `sql.functions.format_string` | Function | `(format: str, cols: ColumnOrName) -> Column` |
| `sql.functions.from_csv` | Function | `(col: ColumnOrName, schema: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.from_json` | Function | `(col: ColumnOrName, schema: Union[ArrayType, StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.from_unixtime` | Function | `(timestamp: ColumnOrName, format: str) -> Column` |
| `sql.functions.from_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.functions.from_xml` | Function | `(col: ColumnOrName, schema: Union[StructType, Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.functools` | Object | `` |
| `sql.functions.get` | Function | `(col: ColumnOrName, index: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.get_json_object` | Function | `(col: ColumnOrName, path: str) -> Column` |
| `sql.functions.getbit` | Function | `(col: ColumnOrName, pos: ColumnOrName) -> Column` |
| `sql.functions.greatest` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.grouping` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.grouping_id` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.hash` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.hex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.histogram_numeric` | Function | `(col: ColumnOrName, nBins: Column) -> Column` |
| `sql.functions.hll_sketch_agg` | Function | `(col: ColumnOrName, lgConfigK: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.hll_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.hll_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, allowDifferentLgConfigK: Optional[bool]) -> Column` |
| `sql.functions.hll_union_agg` | Function | `(col: ColumnOrName, allowDifferentLgConfigK: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.hour` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.hours` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.hypot` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.ifnull` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.ilike` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.functions.initcap` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.inline` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.inline_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.input_file_block_length` | Function | `() -> Column` |
| `sql.functions.input_file_block_start` | Function | `() -> Column` |
| `sql.functions.input_file_name` | Function | `() -> Column` |
| `sql.functions.inspect` | Object | `` |
| `sql.functions.instr` | Function | `(str: ColumnOrName, substr: Union[Column, str]) -> Column` |
| `sql.functions.is_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.is_variant_null` | Function | `(v: ColumnOrName) -> Column` |
| `sql.functions.isnan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.isnotnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.isnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.java_method` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.json_array_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.json_object_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.json_tuple` | Function | `(col: ColumnOrName, fields: str) -> Column` |
| `sql.functions.kll_sketch_agg_bigint` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.kll_sketch_agg_double` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.kll_sketch_agg_float` | Function | `(col: ColumnOrName, k: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.kll_sketch_get_n_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_n_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_n_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_quantile_bigint` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_quantile_double` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_quantile_float` | Function | `(sketch: ColumnOrName, rank: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_rank_bigint` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_rank_double` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_get_rank_float` | Function | `(sketch: ColumnOrName, quantile: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_merge_bigint` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_merge_double` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_merge_float` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_to_string_bigint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_to_string_double` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.kll_sketch_to_string_float` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.kurtosis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.lag` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.functions.last` | Function | `(col: ColumnOrName, ignorenulls: bool) -> Column` |
| `sql.functions.last_day` | Function | `(date: ColumnOrName) -> Column` |
| `sql.functions.last_value` | Function | `(col: ColumnOrName, ignoreNulls: Optional[Union[bool, Column]]) -> Column` |
| `sql.functions.lcase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.lead` | Function | `(col: ColumnOrName, offset: int, default: Optional[Any]) -> Column` |
| `sql.functions.least` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.left` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.functions.length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.levenshtein` | Function | `(left: ColumnOrName, right: ColumnOrName, threshold: Optional[int]) -> Column` |
| `sql.functions.like` | Function | `(str: ColumnOrName, pattern: ColumnOrName, escapeChar: Optional[Column]) -> Column` |
| `sql.functions.listagg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.listagg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.lit` | Function | `(col: Any) -> Column` |
| `sql.functions.ln` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.localtimestamp` | Function | `() -> Column` |
| `sql.functions.locate` | Function | `(substr: str, str: ColumnOrName, pos: int) -> Column` |
| `sql.functions.log` | Function | `(arg1: Union[ColumnOrName, float], arg2: Optional[ColumnOrName]) -> Column` |
| `sql.functions.log10` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.log1p` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.log2` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.lower` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.lpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.functions.ltrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.make_date` | Function | `(year: ColumnOrName, month: ColumnOrName, day: ColumnOrName) -> Column` |
| `sql.functions.make_dt_interval` | Function | `(days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.functions.make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.functions.make_time` | Function | `(hour: ColumnOrName, minute: ColumnOrName, second: ColumnOrName) -> Column` |
| `sql.functions.make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.functions.make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.make_valid_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.make_ym_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName]) -> Column` |
| `sql.functions.map_concat` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.map_contains_key` | Function | `(col: ColumnOrName, value: Any) -> Column` |
| `sql.functions.map_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.map_filter` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.functions.map_from_arrays` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.map_from_entries` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.map_keys` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.map_values` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.map_zip_with` | Function | `(col1: ColumnOrName, col2: ColumnOrName, f: Callable[[Column, Column, Column], Column]) -> Column` |
| `sql.functions.mask` | Function | `(col: ColumnOrName, upperChar: Optional[ColumnOrName], lowerChar: Optional[ColumnOrName], digitChar: Optional[ColumnOrName], otherChar: Optional[ColumnOrName]) -> Column` |
| `sql.functions.max` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.max_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.functions.md5` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.mean` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.median` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.min` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.min_by` | Function | `(col: ColumnOrName, ord: ColumnOrName) -> Column` |
| `sql.functions.minute` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.mode` | Function | `(col: ColumnOrName, deterministic: bool) -> Column` |
| `sql.functions.monotonically_increasing_id` | Function | `() -> Column` |
| `sql.functions.month` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.monthname` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.months` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.months_between` | Function | `(date1: ColumnOrName, date2: ColumnOrName, roundOff: bool) -> Column` |
| `sql.functions.named_struct` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.nanvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.negate` | Object | `` |
| `sql.functions.negative` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.next_day` | Function | `(date: ColumnOrName, dayOfWeek: str) -> Column` |
| `sql.functions.now` | Function | `() -> Column` |
| `sql.functions.nth_value` | Function | `(col: ColumnOrName, offset: int, ignoreNulls: Optional[bool]) -> Column` |
| `sql.functions.ntile` | Function | `(n: int) -> Column` |
| `sql.functions.nullif` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.nullifzero` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.nvl` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.nvl2` | Function | `(col1: ColumnOrName, col2: ColumnOrName, col3: ColumnOrName) -> Column` |
| `sql.functions.octet_length` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.overlay` | Function | `(src: ColumnOrName, replace: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.overload` | Object | `` |
| `sql.functions.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `sql.functions.parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.functions.partitioning.Column` | Class | `(...)` |
| `sql.functions.partitioning.ColumnOrName` | Object | `` |
| `sql.functions.partitioning.PySparkTypeError` | Class | `(...)` |
| `sql.functions.partitioning.bucket` | Function | `(numBuckets: Union[Column, int], col: ColumnOrName) -> Column` |
| `sql.functions.partitioning.days` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.partitioning.hours` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.partitioning.months` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.partitioning.years` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.percent_rank` | Function | `() -> Column` |
| `sql.functions.percentile` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], frequency: Union[Column, int]) -> Column` |
| `sql.functions.percentile_approx` | Function | `(col: ColumnOrName, percentage: Union[Column, float, Sequence[float], Tuple[float, ...]], accuracy: Union[Column, int]) -> Column` |
| `sql.functions.pi` | Function | `() -> Column` |
| `sql.functions.pmod` | Function | `(dividend: Union[ColumnOrName, float], divisor: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.posexplode` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.posexplode_outer` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.position` | Function | `(substr: ColumnOrName, str: ColumnOrName, start: Optional[ColumnOrName]) -> Column` |
| `sql.functions.positive` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.pow` | Function | `(col1: Union[ColumnOrName, float], col2: Union[ColumnOrName, float]) -> Column` |
| `sql.functions.power` | Object | `` |
| `sql.functions.printf` | Function | `(format: ColumnOrName, cols: ColumnOrName) -> Column` |
| `sql.functions.product` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.quarter` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.quote` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.radians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.raise_error` | Function | `(errMsg: Union[Column, str]) -> Column` |
| `sql.functions.rand` | Function | `(seed: Optional[int]) -> Column` |
| `sql.functions.randn` | Function | `(seed: Optional[int]) -> Column` |
| `sql.functions.random` | Object | `` |
| `sql.functions.randstr` | Function | `(length: Union[Column, int], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.rank` | Function | `() -> Column` |
| `sql.functions.reduce` | Function | `(col: ColumnOrName, initialValue: ColumnOrName, merge: Callable[[Column, Column], Column], finish: Optional[Callable[[Column], Column]]) -> Column` |
| `sql.functions.reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.regexp` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.regexp_count` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.regexp_extract` | Function | `(str: ColumnOrName, pattern: str, idx: int) -> Column` |
| `sql.functions.regexp_extract_all` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.regexp_instr` | Function | `(str: ColumnOrName, regexp: ColumnOrName, idx: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.regexp_like` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.regexp_replace` | Function | `(string: ColumnOrName, pattern: Union[str, Column], replacement: Union[str, Column]) -> Column` |
| `sql.functions.regexp_substr` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.regr_avgx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_avgy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_count` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_intercept` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_r2` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_slope` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_sxx` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_sxy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.regr_syy` | Function | `(y: ColumnOrName, x: ColumnOrName) -> Column` |
| `sql.functions.repeat` | Function | `(col: ColumnOrName, n: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.replace` | Function | `(src: ColumnOrName, search: ColumnOrName, replace: Optional[ColumnOrName]) -> Column` |
| `sql.functions.reverse` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.right` | Function | `(str: ColumnOrName, len: ColumnOrName) -> Column` |
| `sql.functions.rint` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.rlike` | Function | `(str: ColumnOrName, regexp: ColumnOrName) -> Column` |
| `sql.functions.round` | Function | `(col: ColumnOrName, scale: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.row_number` | Function | `() -> Column` |
| `sql.functions.rpad` | Function | `(col: ColumnOrName, len: Union[Column, int], pad: Union[Column, str]) -> Column` |
| `sql.functions.rtrim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.schema_of_csv` | Function | `(csv: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.schema_of_json` | Function | `(json: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.schema_of_variant` | Function | `(v: ColumnOrName) -> Column` |
| `sql.functions.schema_of_variant_agg` | Function | `(v: ColumnOrName) -> Column` |
| `sql.functions.schema_of_xml` | Function | `(xml: Union[Column, str], options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.sec` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.second` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sentences` | Function | `(string: ColumnOrName, language: Optional[ColumnOrName], country: Optional[ColumnOrName]) -> Column` |
| `sql.functions.sequence` | Function | `(start: ColumnOrName, stop: ColumnOrName, step: Optional[ColumnOrName]) -> Column` |
| `sql.functions.session_user` | Function | `() -> Column` |
| `sql.functions.session_window` | Function | `(timeColumn: ColumnOrName, gapDuration: Union[Column, str]) -> Column` |
| `sql.functions.sha` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sha1` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sha2` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.shiftLeft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.shiftRight` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.shiftRightUnsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.shiftleft` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.shiftright` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.shiftrightunsigned` | Function | `(col: ColumnOrName, numBits: int) -> Column` |
| `sql.functions.shuffle` | Function | `(col: ColumnOrName, seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.sign` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.signum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sin` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sinh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.size` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.skewness` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.slice` | Function | `(x: ColumnOrName, start: Union[ColumnOrName, int], length: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.some` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sort_array` | Function | `(col: ColumnOrName, asc: bool) -> Column` |
| `sql.functions.soundex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.spark_partition_id` | Function | `() -> Column` |
| `sql.functions.split` | Function | `(str: ColumnOrName, pattern: Union[Column, str], limit: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.split_part` | Function | `(src: ColumnOrName, delimiter: ColumnOrName, partNum: ColumnOrName) -> Column` |
| `sql.functions.sqrt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.st_asbinary` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.functions.st_geogfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.functions.st_geomfromwkb` | Function | `(wkb: ColumnOrName) -> Column` |
| `sql.functions.st_setsrid` | Function | `(geo: ColumnOrName, srid: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.st_srid` | Function | `(geo: ColumnOrName) -> Column` |
| `sql.functions.stack` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.startswith` | Function | `(str: ColumnOrName, prefix: ColumnOrName) -> Column` |
| `sql.functions.std` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.stddev` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.stddev_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.stddev_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.str_to_map` | Function | `(text: ColumnOrName, pairDelim: Optional[ColumnOrName], keyValueDelim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.string_agg` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.string_agg_distinct` | Function | `(col: ColumnOrName, delimiter: Optional[Union[Column, str, bytes]]) -> Column` |
| `sql.functions.struct` | Function | `(cols: Union[ColumnOrName, Union[Sequence[ColumnOrName], Tuple[ColumnOrName, ...]]]) -> Column` |
| `sql.functions.substr` | Function | `(str: ColumnOrName, pos: ColumnOrName, len: Optional[ColumnOrName]) -> Column` |
| `sql.functions.substring` | Function | `(str: ColumnOrName, pos: Union[ColumnOrName, int], len: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.substring_index` | Function | `(str: ColumnOrName, delim: str, count: int) -> Column` |
| `sql.functions.sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sumDistinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sum_distinct` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.sys` | Object | `` |
| `sql.functions.tan` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.tanh` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.theta_difference` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.theta_intersection` | Function | `(col1: ColumnOrName, col2: ColumnOrName) -> Column` |
| `sql.functions.theta_intersection_agg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.theta_sketch_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.theta_sketch_estimate` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.theta_union` | Function | `(col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.theta_union_agg` | Function | `(col: ColumnOrName, lgNomEntries: Optional[Union[int, Column]]) -> Column` |
| `sql.functions.time_diff` | Function | `(unit: ColumnOrName, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.functions.time_trunc` | Function | `(unit: ColumnOrName, time: ColumnOrName) -> Column` |
| `sql.functions.timestamp_add` | Function | `(unit: str, quantity: ColumnOrName, ts: ColumnOrName) -> Column` |
| `sql.functions.timestamp_diff` | Function | `(unit: str, start: ColumnOrName, end: ColumnOrName) -> Column` |
| `sql.functions.timestamp_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.timestamp_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.timestamp_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.toDegrees` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.toRadians` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.to_char` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.to_csv` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.functions.to_json` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.to_timestamp` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.functions.to_timestamp_ltz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.to_timestamp_ntz` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.to_unix_timestamp` | Function | `(timestamp: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.to_utc_timestamp` | Function | `(timestamp: ColumnOrName, tz: Union[Column, str]) -> Column` |
| `sql.functions.to_varchar` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.to_variant_object` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.to_xml` | Function | `(col: ColumnOrName, options: Optional[Mapping[str, str]]) -> Column` |
| `sql.functions.transform` | Function | `(col: ColumnOrName, f: Union[Callable[[Column], Column], Callable[[Column, Column], Column]]) -> Column` |
| `sql.functions.transform_keys` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.functions.transform_values` | Function | `(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.functions.translate` | Function | `(srcCol: ColumnOrName, matching: str, replace: str) -> Column` |
| `sql.functions.trim` | Function | `(col: ColumnOrName, trim: Optional[ColumnOrName]) -> Column` |
| `sql.functions.trunc` | Function | `(date: ColumnOrName, format: str) -> Column` |
| `sql.functions.try_add` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.try_aes_decrypt` | Function | `(input: ColumnOrName, key: ColumnOrName, mode: Optional[ColumnOrName], padding: Optional[ColumnOrName], aad: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_avg` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.try_divide` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.try_element_at` | Function | `(col: ColumnOrName, extraction: ColumnOrName) -> Column` |
| `sql.functions.try_make_interval` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], weeks: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_make_timestamp` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], timezone: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_make_timestamp_ltz` | Function | `(years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_make_timestamp_ntz` | Function | `(years: Optional[ColumnOrName], months: Optional[ColumnOrName], days: Optional[ColumnOrName], hours: Optional[ColumnOrName], mins: Optional[ColumnOrName], secs: Optional[ColumnOrName], date: Optional[ColumnOrName], time: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_mod` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.try_multiply` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.try_parse_json` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.try_parse_url` | Function | `(url: ColumnOrName, partToExtract: ColumnOrName, key: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_reflect` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.try_subtract` | Function | `(left: ColumnOrName, right: ColumnOrName) -> Column` |
| `sql.functions.try_sum` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.try_to_binary` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_to_date` | Function | `(col: ColumnOrName, format: Optional[str]) -> Column` |
| `sql.functions.try_to_number` | Function | `(col: ColumnOrName, format: ColumnOrName) -> Column` |
| `sql.functions.try_to_time` | Function | `(str: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_to_timestamp` | Function | `(col: ColumnOrName, format: Optional[ColumnOrName]) -> Column` |
| `sql.functions.try_url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.try_validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.try_variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.functions.typeof` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.ucase` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.udf` | Function | `(f: Optional[Union[Callable[..., Any], DataTypeOrString]], returnType: DataTypeOrString, useArrow: Optional[bool]) -> Union[UserDefinedFunctionLike, Callable[[Callable[..., Any]], UserDefinedFunctionLike]]` |
| `sql.functions.udtf` | Function | `(cls: Optional[Type], returnType: Optional[Union[StructType, str]], useArrow: Optional[bool]) -> Union[UserDefinedTableFunction, Callable[[Type], UserDefinedTableFunction]]` |
| `sql.functions.unbase64` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.unhex` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.uniform` | Function | `(min: Union[Column, int, float], max: Union[Column, int, float], seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.unix_date` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.unix_micros` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.unix_millis` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.unix_seconds` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.unix_timestamp` | Function | `(timestamp: Optional[ColumnOrName], format: str) -> Column` |
| `sql.functions.unwrap_udt` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.upper` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.url_decode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.url_encode` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.user` | Function | `() -> Column` |
| `sql.functions.uuid` | Function | `(seed: Optional[Union[Column, int]]) -> Column` |
| `sql.functions.validate_utf8` | Function | `(str: ColumnOrName) -> Column` |
| `sql.functions.var_pop` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.var_samp` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.variance` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.variant_get` | Function | `(v: ColumnOrName, path: Union[Column, str], targetType: str) -> Column` |
| `sql.functions.version` | Function | `() -> Column` |
| `sql.functions.warnings` | Object | `` |
| `sql.functions.weekday` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.weekofyear` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.when` | Function | `(condition: Column, value: Any) -> Column` |
| `sql.functions.width_bucket` | Function | `(v: ColumnOrName, min: ColumnOrName, max: ColumnOrName, numBucket: Union[ColumnOrName, int]) -> Column` |
| `sql.functions.window` | Function | `(timeColumn: ColumnOrName, windowDuration: str, slideDuration: Optional[str], startTime: Optional[str]) -> Column` |
| `sql.functions.window_time` | Function | `(windowColumn: ColumnOrName) -> Column` |
| `sql.functions.xpath` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_boolean` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_double` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_float` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_int` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_long` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_number` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_short` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xpath_string` | Function | `(xml: ColumnOrName, path: ColumnOrName) -> Column` |
| `sql.functions.xxhash64` | Function | `(cols: ColumnOrName) -> Column` |
| `sql.functions.year` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.years` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.zeroifnull` | Function | `(col: ColumnOrName) -> Column` |
| `sql.functions.zip_with` | Function | `(left: ColumnOrName, right: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column` |
| `sql.geo_utils.CartesianSpatialReferenceSystemMapper` | Class | `(...)` |
| `sql.geo_utils.GeographicSpatialReferenceSystemMapper` | Class | `(...)` |
| `sql.geo_utils.SpatialReferenceSystemCache` | Class | `()` |
| `sql.geo_utils.SpatialReferenceSystemInformation` | Class | `(srid: int, string_id: str, is_geographic: bool)` |
| `sql.group.Column` | Class | `(...)` |
| `sql.group.DataFrame` | Class | `(...)` |
| `sql.group.GroupedData` | Class | `(jgd: JavaObject, df: DataFrame)` |
| `sql.group.LiteralType` | Object | `` |
| `sql.group.PandasGroupedOpsMixin` | Class | `(...)` |
| `sql.group.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.group.df_varargs_api` | Function | `(f: Callable[..., DataFrame]) -> Callable[..., DataFrame]` |
| `sql.group.dfapi` | Function | `(f: Callable[..., DataFrame]) -> Callable[..., DataFrame]` |
| `sql.internal.Column` | Class | `(...)` |
| `sql.internal.ColumnOrName` | Object | `` |
| `sql.internal.F` | Object | `` |
| `sql.internal.InternalFunction` | Class | `(...)` |
| `sql.internal.is_remote` | Function | `() -> bool` |
| `sql.is_remote` | Function | `() -> bool` |
| `sql.merge.Column` | Class | `(...)` |
| `sql.merge.DataFrame` | Class | `(...)` |
| `sql.merge.MergeIntoWriter` | Class | `(df: DataFrame, table: str, condition: Column)` |
| `sql.merge.to_scala_map` | Function | `(jvm: JVMView, dic: Dict) -> JavaObject` |
| `sql.metrics.CollectedMetrics` | Class | `(metrics: List[PlanMetrics])` |
| `sql.metrics.ExecutionInfo` | Class | `(metrics: Optional[list[PlanMetrics]], obs: Optional[Sequence[ObservedMetrics]])` |
| `sql.metrics.MetricValue` | Class | `(name: str, value: Union[int, float], type: str)` |
| `sql.metrics.ObservedMetrics` | Class | `(...)` |
| `sql.metrics.PlanMetrics` | Class | `(name: str, id: int, parent: int, metrics: List[MetricValue])` |
| `sql.metrics.PySparkValueError` | Class | `(...)` |
| `sql.observation.CPickleSerializer` | Object | `` |
| `sql.observation.Column` | Class | `(...)` |
| `sql.observation.DataFrame` | Class | `(...)` |
| `sql.observation.Observation` | Class | `(name: Optional[str])` |
| `sql.observation.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.observation.PySparkTypeError` | Class | `(...)` |
| `sql.observation.PySparkValueError` | Class | `(...)` |
| `sql.observation.Row` | Class | `(...)` |
| `sql.observation.is_remote` | Function | `() -> bool` |
| `sql.pandas.conversion.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.pandas.conversion.ArrowCollectSerializer` | Class | `()` |
| `sql.pandas.conversion.DataFrame` | Class | `(...)` |
| `sql.pandas.conversion.DataType` | Class | `(...)` |
| `sql.pandas.conversion.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `sql.pandas.conversion.PandasConversionMixin` | Class | `(...)` |
| `sql.pandas.conversion.PandasDataFrameLike` | Object | `` |
| `sql.pandas.conversion.PySparkTypeError` | Class | `(...)` |
| `sql.pandas.conversion.PySparkValueError` | Class | `(...)` |
| `sql.pandas.conversion.SCCallSiteSync` | Class | `(sc)` |
| `sql.pandas.conversion.SparkConversionMixin` | Class | `(...)` |
| `sql.pandas.conversion.StringType` | Class | `(collation: str)` |
| `sql.pandas.conversion.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.pandas.conversion.TimestampType` | Class | `(...)` |
| `sql.pandas.conversion.is_timestamp_ntz_preferred` | Function | `() -> bool` |
| `sql.pandas.conversion.unwrap_spark_exception` | Function | `() -> Iterator[Any]` |
| `sql.pandas.functions.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.pandas.functions.ArrowGroupedAggUDFType` | Object | `` |
| `sql.pandas.functions.ArrowScalarIterFunction` | Object | `` |
| `sql.pandas.functions.ArrowScalarIterUDFType` | Object | `` |
| `sql.pandas.functions.ArrowScalarToScalarFunction` | Object | `` |
| `sql.pandas.functions.ArrowScalarUDFType` | Object | `` |
| `sql.pandas.functions.ArrowUDFType` | Class | `(...)` |
| `sql.pandas.functions.AtomicDataTypeOrString` | Object | `` |
| `sql.pandas.functions.DataType` | Class | `(...)` |
| `sql.pandas.functions.DataTypeOrString` | Object | `` |
| `sql.pandas.functions.GroupedMapPandasUserDefinedFunction` | Object | `` |
| `sql.pandas.functions.PandasGroupedAggFunction` | Object | `` |
| `sql.pandas.functions.PandasGroupedAggUDFType` | Object | `` |
| `sql.pandas.functions.PandasGroupedMapFunction` | Object | `` |
| `sql.pandas.functions.PandasGroupedMapIterUDFType` | Object | `` |
| `sql.pandas.functions.PandasGroupedMapUDFType` | Object | `` |
| `sql.pandas.functions.PandasScalarIterFunction` | Object | `` |
| `sql.pandas.functions.PandasScalarIterUDFType` | Object | `` |
| `sql.pandas.functions.PandasScalarToScalarFunction` | Object | `` |
| `sql.pandas.functions.PandasScalarToStructFunction` | Object | `` |
| `sql.pandas.functions.PandasScalarUDFType` | Object | `` |
| `sql.pandas.functions.PandasUDFType` | Class | `(...)` |
| `sql.pandas.functions.PySparkTypeError` | Class | `(...)` |
| `sql.pandas.functions.PySparkValueError` | Class | `(...)` |
| `sql.pandas.functions.PythonEvalType` | Class | `(...)` |
| `sql.pandas.functions.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.pandas.functions.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.pandas.functions.arrow_udf` | Function | `(f: ArrowScalarToScalarFunction \| DataTypeOrString \| ArrowScalarIterFunction \| (AtomicDataTypeOrString \| ArrayType), returnType: DataTypeOrString \| ArrowScalarUDFType \| (AtomicDataTypeOrString \| ArrayType) \| ArrowScalarIterUDFType, functionType: ArrowScalarUDFType \| ArrowScalarIterUDFType) -> UserDefinedFunctionLike \| Callable[[ArrowScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[ArrowScalarIterFunction], UserDefinedFunctionLike]` |
| `sql.pandas.functions.infer_eval_type` | Function | `(sig: Signature, type_hints: Dict[str, Any], kind: str) -> Union[PandasScalarUDFType, PandasScalarIterUDFType, PandasGroupedAggUDFType, ArrowScalarUDFType, ArrowScalarIterUDFType, ArrowGroupedAggUDFType]` |
| `sql.pandas.functions.is_remote` | Function | `() -> bool` |
| `sql.pandas.functions.pandas_udf` | Function | `(f: PandasScalarToScalarFunction \| (AtomicDataTypeOrString \| ArrayType) \| PandasScalarToStructFunction \| (StructType \| str) \| PandasScalarIterFunction \| PandasGroupedMapFunction \| PandasGroupedAggFunction, returnType: AtomicDataTypeOrString \| ArrayType \| PandasScalarUDFType \| (StructType \| str) \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType, functionType: PandasScalarUDFType \| PandasScalarIterUDFType \| (PandasGroupedMapUDFType \| PandasGroupedMapIterUDFType) \| PandasGroupedAggUDFType) -> UserDefinedFunctionLike \| Callable[[PandasScalarToScalarFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarToStructFunction], UserDefinedFunctionLike] \| Callable[[PandasScalarIterFunction], UserDefinedFunctionLike] \| GroupedMapPandasUserDefinedFunction \| Callable[[PandasGroupedMapFunction], GroupedMapPandasUserDefinedFunction] \| Callable[[PandasGroupedAggFunction], UserDefinedFunctionLike]` |
| `sql.pandas.functions.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.pandas.functions.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.pandas.functions.since` | Function | `(version: Union[str, float]) -> Callable[[_F], _F]` |
| `sql.pandas.functions.vectorized_udf` | Function | `(f, returnType, functionType, kind: str)` |
| `sql.pandas.group_ops.ArrowCogroupedMapFunction` | Object | `` |
| `sql.pandas.group_ops.ArrowGroupedMapFunction` | Object | `` |
| `sql.pandas.group_ops.Column` | Class | `(...)` |
| `sql.pandas.group_ops.DataFrame` | Class | `(...)` |
| `sql.pandas.group_ops.GroupStateTimeout` | Class | `(...)` |
| `sql.pandas.group_ops.GroupedData` | Class | `(jgd: JavaObject, df: DataFrame)` |
| `sql.pandas.group_ops.GroupedMapPandasUserDefinedFunction` | Object | `` |
| `sql.pandas.group_ops.PandasCogroupedMapFunction` | Object | `` |
| `sql.pandas.group_ops.PandasCogroupedOps` | Class | `(gd1: GroupedData, gd2: GroupedData)` |
| `sql.pandas.group_ops.PandasGroupedMapFunction` | Object | `` |
| `sql.pandas.group_ops.PandasGroupedMapFunctionWithState` | Object | `` |
| `sql.pandas.group_ops.PandasGroupedOpsMixin` | Class | `(...)` |
| `sql.pandas.group_ops.PySparkTypeError` | Class | `(...)` |
| `sql.pandas.group_ops.PythonEvalType` | Class | `(...)` |
| `sql.pandas.group_ops.StatefulProcessor` | Class | `(...)` |
| `sql.pandas.group_ops.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.pandas.group_ops.infer_group_arrow_eval_type_from_func` | Function | `(f: ArrowGroupedMapFunction) -> Optional[Union[ArrowGroupedMapUDFType, ArrowGroupedMapIterUDFType]]` |
| `sql.pandas.map_ops.ArrowMapIterFunction` | Object | `` |
| `sql.pandas.map_ops.DataFrame` | Class | `(...)` |
| `sql.pandas.map_ops.ExecutorResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, ExecutorResourceRequest]])` |
| `sql.pandas.map_ops.PandasMapIterFunction` | Object | `` |
| `sql.pandas.map_ops.PandasMapOpsMixin` | Class | `(...)` |
| `sql.pandas.map_ops.PythonEvalType` | Class | `(...)` |
| `sql.pandas.map_ops.ResourceProfile` | Class | `(_java_resource_profile: Optional[JavaObject], _exec_req: Optional[Dict[str, ExecutorResourceRequest]], _task_req: Optional[Dict[str, TaskResourceRequest]])` |
| `sql.pandas.map_ops.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.pandas.map_ops.TaskResourceRequests` | Class | `(_jvm: Optional[JVMView], _requests: Optional[Dict[str, TaskResourceRequest]])` |
| `sql.pandas.serializers.ApplyInPandasWithStateSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, state_object_schema, arrow_max_records_per_batch, prefers_large_var_types, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.ArrowBatchUDFSerializer` | Class | `(timezone, safecheck, input_types, int_to_decimal_coercion_enabled, binary_as_bytes)` |
| `sql.pandas.serializers.ArrowCollectSerializer` | Class | `()` |
| `sql.pandas.serializers.ArrowStreamAggArrowUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_cast)` |
| `sql.pandas.serializers.ArrowStreamArrowUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_cast)` |
| `sql.pandas.serializers.ArrowStreamArrowUDTFSerializer` | Class | `(table_arg_offsets)` |
| `sql.pandas.serializers.ArrowStreamGroupUDFSerializer` | Class | `(assign_cols_by_name)` |
| `sql.pandas.serializers.ArrowStreamPandasSerializer` | Class | `(timezone, safecheck, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.ArrowStreamPandasUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, df_for_struct, struct_in_pandas, ndarray_as_list, arrow_cast, input_types, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.ArrowStreamPandasUDTFSerializer` | Class | `(timezone, safecheck, input_types, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.ArrowStreamSerializer` | Class | `(...)` |
| `sql.pandas.serializers.ArrowStreamUDFSerializer` | Class | `(...)` |
| `sql.pandas.serializers.ArrowStreamUDTFSerializer` | Class | `(...)` |
| `sql.pandas.serializers.ArrowTableToRowsConversion` | Class | `(...)` |
| `sql.pandas.serializers.BinaryType` | Class | `(...)` |
| `sql.pandas.serializers.CPickleSerializer` | Object | `` |
| `sql.pandas.serializers.CogroupArrowUDFSerializer` | Class | `(assign_cols_by_name)` |
| `sql.pandas.serializers.CogroupPandasUDFSerializer` | Class | `(...)` |
| `sql.pandas.serializers.DataType` | Class | `(...)` |
| `sql.pandas.serializers.GroupArrowUDFSerializer` | Class | `(assign_cols_by_name)` |
| `sql.pandas.serializers.GroupPandasIterUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.GroupPandasUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.IntegerType` | Class | `(...)` |
| `sql.pandas.serializers.LocalDataToArrowConversion` | Class | `(...)` |
| `sql.pandas.serializers.LongType` | Class | `(...)` |
| `sql.pandas.serializers.PySparkRuntimeError` | Class | `(...)` |
| `sql.pandas.serializers.PySparkTypeError` | Class | `(...)` |
| `sql.pandas.serializers.PySparkValueError` | Class | `(...)` |
| `sql.pandas.serializers.Row` | Class | `(...)` |
| `sql.pandas.serializers.Serializer` | Class | `(...)` |
| `sql.pandas.serializers.SpecialLengths` | Class | `(...)` |
| `sql.pandas.serializers.StringType` | Class | `(collation: str)` |
| `sql.pandas.serializers.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `sql.pandas.serializers.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.pandas.serializers.TransformWithStateInPandasInitStateSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_max_records_per_batch, arrow_max_bytes_per_batch, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.TransformWithStateInPandasSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_max_records_per_batch, arrow_max_bytes_per_batch, int_to_decimal_coercion_enabled)` |
| `sql.pandas.serializers.TransformWithStateInPySparkRowInitStateSerializer` | Class | `(arrow_max_records_per_batch)` |
| `sql.pandas.serializers.TransformWithStateInPySparkRowSerializer` | Class | `(arrow_max_records_per_batch)` |
| `sql.pandas.serializers.UTF8Deserializer` | Class | `(use_unicode)` |
| `sql.pandas.serializers.from_arrow_type` | Function | `(at: pa.DataType, prefer_timestamp_ntz: bool) -> DataType` |
| `sql.pandas.serializers.is_variant` | Function | `(at: pa.DataType) -> bool` |
| `sql.pandas.serializers.read_int` | Function | `(stream)` |
| `sql.pandas.serializers.to_arrow_type` | Function | `(dt: DataType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.DataType` |
| `sql.pandas.serializers.write_int` | Function | `(value, stream)` |
| `sql.pandas.typehints.ArrowGroupedAggUDFType` | Object | `` |
| `sql.pandas.typehints.ArrowGroupedMapFunction` | Object | `` |
| `sql.pandas.typehints.ArrowGroupedMapIterUDFType` | Object | `` |
| `sql.pandas.typehints.ArrowGroupedMapUDFType` | Object | `` |
| `sql.pandas.typehints.ArrowScalarIterUDFType` | Object | `` |
| `sql.pandas.typehints.ArrowScalarUDFType` | Object | `` |
| `sql.pandas.typehints.PandasGroupedAggUDFType` | Object | `` |
| `sql.pandas.typehints.PandasGroupedMapFunction` | Object | `` |
| `sql.pandas.typehints.PandasGroupedMapIterUDFType` | Object | `` |
| `sql.pandas.typehints.PandasGroupedMapUDFType` | Object | `` |
| `sql.pandas.typehints.PandasScalarIterUDFType` | Object | `` |
| `sql.pandas.typehints.PandasScalarUDFType` | Object | `` |
| `sql.pandas.typehints.PySparkNotImplementedError` | Class | `(...)` |
| `sql.pandas.typehints.PySparkValueError` | Class | `(...)` |
| `sql.pandas.typehints.check_iterator_annotation` | Function | `(annotation: Any, parameter_check_func: Optional[Callable[[Any], bool]]) -> bool` |
| `sql.pandas.typehints.check_tuple_annotation` | Function | `(annotation: Any, parameter_check_func: Optional[Callable[[Any], bool]]) -> bool` |
| `sql.pandas.typehints.check_union_annotation` | Function | `(annotation: Any, parameter_check_func: Optional[Callable[[Any], bool]]) -> bool` |
| `sql.pandas.typehints.infer_arrow_eval_type` | Function | `(sig: Signature, type_hints: Dict[str, Any]) -> Optional[Union[ArrowScalarUDFType, ArrowScalarIterUDFType, ArrowGroupedAggUDFType]]` |
| `sql.pandas.typehints.infer_eval_type` | Function | `(sig: Signature, type_hints: Dict[str, Any], kind: str) -> Union[PandasScalarUDFType, PandasScalarIterUDFType, PandasGroupedAggUDFType, ArrowScalarUDFType, ArrowScalarIterUDFType, ArrowGroupedAggUDFType]` |
| `sql.pandas.typehints.infer_eval_type_for_udf` | Function | `(f) -> Optional[Union[PandasScalarUDFType, PandasScalarIterUDFType, PandasGroupedAggUDFType, ArrowScalarUDFType, ArrowScalarIterUDFType, ArrowGroupedAggUDFType]]` |
| `sql.pandas.typehints.infer_group_arrow_eval_type` | Function | `(sig: Signature, type_hints: Dict[str, Any]) -> Optional[Union[ArrowGroupedMapUDFType, ArrowGroupedMapIterUDFType]]` |
| `sql.pandas.typehints.infer_group_arrow_eval_type_from_func` | Function | `(f: ArrowGroupedMapFunction) -> Optional[Union[ArrowGroupedMapUDFType, ArrowGroupedMapIterUDFType]]` |
| `sql.pandas.typehints.infer_group_pandas_eval_type` | Function | `(sig: Signature, type_hints: Dict[str, Any]) -> Optional[Union[PandasGroupedMapUDFType, PandasGroupedMapIterUDFType]]` |
| `sql.pandas.typehints.infer_group_pandas_eval_type_from_func` | Function | `(f: PandasGroupedMapFunction) -> Optional[Union[PandasGroupedMapUDFType, PandasGroupedMapIterUDFType]]` |
| `sql.pandas.typehints.infer_pandas_eval_type` | Function | `(sig: Signature, type_hints: Dict[str, Any]) -> Optional[Union[PandasScalarUDFType, PandasScalarIterUDFType, PandasGroupedAggUDFType]]` |
| `sql.pandas.typehints.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.pandas.typehints.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.pandas.types.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.pandas.types.BinaryType` | Class | `(...)` |
| `sql.pandas.types.BooleanType` | Class | `(...)` |
| `sql.pandas.types.ByteType` | Class | `(...)` |
| `sql.pandas.types.DataType` | Class | `(...)` |
| `sql.pandas.types.DateType` | Class | `(...)` |
| `sql.pandas.types.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `sql.pandas.types.DecimalType` | Class | `(precision: int, scale: int)` |
| `sql.pandas.types.DoubleType` | Class | `(...)` |
| `sql.pandas.types.FloatType` | Class | `(...)` |
| `sql.pandas.types.Geography` | Class | `(wkb: bytes, srid: int)` |
| `sql.pandas.types.GeographyType` | Class | `(srid: int \| str)` |
| `sql.pandas.types.Geometry` | Class | `(wkb: bytes, srid: int)` |
| `sql.pandas.types.GeometryType` | Class | `(srid: int \| str)` |
| `sql.pandas.types.IntegerType` | Class | `(...)` |
| `sql.pandas.types.IntegralType` | Class | `(...)` |
| `sql.pandas.types.LongType` | Class | `(...)` |
| `sql.pandas.types.LooseVersion` | Class | `(vstring: Optional[str])` |
| `sql.pandas.types.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `sql.pandas.types.NullType` | Class | `(...)` |
| `sql.pandas.types.PandasDataFrameLike` | Object | `` |
| `sql.pandas.types.PandasSeriesLike` | Object | `` |
| `sql.pandas.types.PySparkTypeError` | Class | `(...)` |
| `sql.pandas.types.PySparkValueError` | Class | `(...)` |
| `sql.pandas.types.ShortType` | Class | `(...)` |
| `sql.pandas.types.StringType` | Class | `(collation: str)` |
| `sql.pandas.types.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `sql.pandas.types.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.pandas.types.TimeType` | Class | `(precision: int)` |
| `sql.pandas.types.TimestampNTZType` | Class | `(...)` |
| `sql.pandas.types.TimestampType` | Class | `(...)` |
| `sql.pandas.types.UnsupportedOperationException` | Class | `(...)` |
| `sql.pandas.types.UserDefinedType` | Class | `(...)` |
| `sql.pandas.types.VariantType` | Class | `(...)` |
| `sql.pandas.types.VariantVal` | Class | `(value: bytes, metadata: bytes)` |
| `sql.pandas.types.cast` | Object | `` |
| `sql.pandas.types.convert_pandas_using_numpy_type` | Function | `(df: PandasDataFrameLike, schema: StructType) -> PandasDataFrameLike` |
| `sql.pandas.types.from_arrow_schema` | Function | `(arrow_schema: pa.Schema, prefer_timestamp_ntz: bool) -> StructType` |
| `sql.pandas.types.from_arrow_type` | Function | `(at: pa.DataType, prefer_timestamp_ntz: bool) -> DataType` |
| `sql.pandas.types.is_geography` | Function | `(at: pa.DataType) -> bool` |
| `sql.pandas.types.is_geometry` | Function | `(at: pa.DataType) -> bool` |
| `sql.pandas.types.is_variant` | Function | `(at: pa.DataType) -> bool` |
| `sql.pandas.types.to_arrow_schema` | Function | `(schema: StructType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.Schema` |
| `sql.pandas.types.to_arrow_type` | Function | `(dt: DataType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.DataType` |
| `sql.pandas.utils.LooseVersion` | Class | `(vstring: Optional[str])` |
| `sql.pandas.utils.PySparkImportError` | Class | `(...)` |
| `sql.pandas.utils.PySparkRuntimeError` | Class | `(...)` |
| `sql.pandas.utils.require_minimum_numpy_version` | Function | `() -> None` |
| `sql.pandas.utils.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.pandas.utils.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.plot.Any` | Object | `` |
| `sql.plot.Column` | Class | `(...)` |
| `sql.plot.F` | Object | `` |
| `sql.plot.List` | Object | `` |
| `sql.plot.ModuleType` | Object | `` |
| `sql.plot.NumpyHelper` | Class | `(...)` |
| `sql.plot.Optional` | Object | `` |
| `sql.plot.PySparkBoxPlotBase` | Class | `(...)` |
| `sql.plot.PySparkHistogramPlotBase` | Class | `(...)` |
| `sql.plot.PySparkKdePlotBase` | Class | `(...)` |
| `sql.plot.PySparkPlotAccessor` | Class | `(data: DataFrame)` |
| `sql.plot.PySparkSampledPlotBase` | Class | `(...)` |
| `sql.plot.PySparkTopNPlotBase` | Class | `(...)` |
| `sql.plot.PySparkValueError` | Class | `(...)` |
| `sql.plot.SF` | Class | `(...)` |
| `sql.plot.Sequence` | Object | `` |
| `sql.plot.TYPE_CHECKING` | Object | `` |
| `sql.plot.Union` | Object | `` |
| `sql.plot.core.Column` | Class | `(...)` |
| `sql.plot.core.DataFrame` | Class | `(...)` |
| `sql.plot.core.F` | Object | `` |
| `sql.plot.core.NumpyHelper` | Class | `(...)` |
| `sql.plot.core.PySparkBoxPlotBase` | Class | `(...)` |
| `sql.plot.core.PySparkHistogramPlotBase` | Class | `(...)` |
| `sql.plot.core.PySparkKdePlotBase` | Class | `(...)` |
| `sql.plot.core.PySparkPlotAccessor` | Class | `(data: DataFrame)` |
| `sql.plot.core.PySparkSampledPlotBase` | Class | `(...)` |
| `sql.plot.core.PySparkTopNPlotBase` | Class | `(...)` |
| `sql.plot.core.PySparkValueError` | Class | `(...)` |
| `sql.plot.core.Row` | Class | `(...)` |
| `sql.plot.core.SF` | Class | `(...)` |
| `sql.plot.core.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.plot.core.require_minimum_plotly_version` | Function | `() -> None` |
| `sql.plot.math` | Object | `` |
| `sql.plot.plotly.DataFrame` | Class | `(...)` |
| `sql.plot.plotly.NumericType` | Class | `(...)` |
| `sql.plot.plotly.PySparkBoxPlotBase` | Class | `(...)` |
| `sql.plot.plotly.PySparkHistogramPlotBase` | Class | `(...)` |
| `sql.plot.plotly.PySparkKdePlotBase` | Class | `(...)` |
| `sql.plot.plotly.PySparkPlotAccessor` | Class | `(data: DataFrame)` |
| `sql.plot.plotly.PySparkTypeError` | Class | `(...)` |
| `sql.plot.plotly.PySparkValueError` | Class | `(...)` |
| `sql.plot.plotly.plot_box` | Function | `(data: DataFrame, kwargs: Any) -> Figure` |
| `sql.plot.plotly.plot_histogram` | Function | `(data: DataFrame, kwargs: Any) -> Figure` |
| `sql.plot.plotly.plot_kde` | Function | `(data: DataFrame, kwargs: Any) -> Figure` |
| `sql.plot.plotly.plot_pie` | Function | `(data: DataFrame, kwargs: Any) -> Figure` |
| `sql.plot.plotly.plot_pyspark` | Function | `(data: DataFrame, kind: str, kwargs: Any) -> Figure` |
| `sql.plot.plotly.process_column_param` | Function | `(column: Optional[Union[str, List[str]]], data: DataFrame) -> List[str]` |
| `sql.plot.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.plot.require_minimum_plotly_version` | Function | `() -> None` |
| `sql.profiler.Accumulator` | Class | `(aid: int, value: T, accum_param: AccumulatorParam[T])` |
| `sql.profiler.AccumulatorParam` | Class | `(...)` |
| `sql.profiler.AccumulatorProfilerCollector` | Class | `()` |
| `sql.profiler.CodeMapDict` | Object | `` |
| `sql.profiler.MemUsageParam` | Class | `(...)` |
| `sql.profiler.MemoryProfiler` | Class | `(ctx: SparkContext)` |
| `sql.profiler.PStatsParam` | Class | `(...)` |
| `sql.profiler.Profile` | Class | `(profiler_collector: ProfilerCollector)` |
| `sql.profiler.ProfileResults` | Object | `` |
| `sql.profiler.ProfileResultsParam` | Object | `` |
| `sql.profiler.ProfilerCollector` | Class | `()` |
| `sql.profiler.PySparkValueError` | Class | `(...)` |
| `sql.profiler.SpecialAccumulatorIds` | Class | `(...)` |
| `sql.profiler.has_memory_profiler` | Object | `` |
| `sql.protobuf.functions.Column` | Class | `(...)` |
| `sql.protobuf.functions.ColumnOrName` | Object | `` |
| `sql.protobuf.functions.from_protobuf` | Function | `(data: ColumnOrName, messageName: str, descFilePath: Optional[str], options: Optional[Dict[str, str]], binaryDescriptorSet: Optional[bytes]) -> Column` |
| `sql.protobuf.functions.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.protobuf.functions.to_protobuf` | Function | `(data: ColumnOrName, messageName: str, descFilePath: Optional[str], options: Optional[Dict[str, str]], binaryDescriptorSet: Optional[bytes]) -> Column` |
| `sql.protobuf.functions.try_remote_protobuf_functions` | Function | `(f: FuncT) -> FuncT` |
| `sql.readwriter.ColumnOrName` | Object | `` |
| `sql.readwriter.DataFrame` | Class | `(...)` |
| `sql.readwriter.DataFrameReader` | Class | `(spark: SparkSession)` |
| `sql.readwriter.DataFrameWriter` | Class | `(df: DataFrame)` |
| `sql.readwriter.DataFrameWriterV2` | Class | `(df: DataFrame, table: str)` |
| `sql.readwriter.OptionUtils` | Class | `(...)` |
| `sql.readwriter.OptionalPrimitiveType` | Object | `` |
| `sql.readwriter.PathOrPaths` | Object | `` |
| `sql.readwriter.PySparkTypeError` | Class | `(...)` |
| `sql.readwriter.PySparkValueError` | Class | `(...)` |
| `sql.readwriter.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `sql.readwriter.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.readwriter.StreamingQuery` | Class | `(jsq: JavaObject)` |
| `sql.readwriter.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.readwriter.TupleOrListOfString` | Object | `` |
| `sql.readwriter.is_remote_only` | Function | `() -> bool` |
| `sql.readwriter.to_str` | Function | `(value: Any) -> Optional[str]` |
| `sql.readwriter.utils` | Object | `` |
| `sql.session.AccumulatorProfilerCollector` | Class | `()` |
| `sql.session.ArrayLike` | Object | `` |
| `sql.session.AtomicType` | Class | `(...)` |
| `sql.session.AtomicValue` | Object | `` |
| `sql.session.Catalog` | Class | `(sparkSession: SparkSession)` |
| `sql.session.DataFrame` | Class | `(...)` |
| `sql.session.DataFrameReader` | Class | `(spark: SparkSession)` |
| `sql.session.DataSourceRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.session.DataStreamReader` | Class | `(spark: SparkSession)` |
| `sql.session.DataType` | Class | `(...)` |
| `sql.session.OptionalPrimitiveType` | Object | `` |
| `sql.session.PandasDataFrameLike` | Object | `` |
| `sql.session.ParentDataFrame` | Class | `(...)` |
| `sql.session.Profile` | Class | `(profiler_collector: ProfilerCollector)` |
| `sql.session.ProgressHandler` | Class | `(...)` |
| `sql.session.PySparkRuntimeError` | Class | `(...)` |
| `sql.session.PySparkTypeError` | Class | `(...)` |
| `sql.session.PySparkValueError` | Class | `(...)` |
| `sql.session.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `sql.session.RowLike` | Object | `` |
| `sql.session.RuntimeConfig` | Class | `(jconf: JavaObject)` |
| `sql.session.SQLStringFormatter` | Class | `(session: SparkSession)` |
| `sql.session.SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `sql.session.SparkConnectClient` | Class | `(connection: Union[str, ChannelBuilder], user_id: Optional[str], channel_options: Optional[List[Tuple[str, Any]]], retry_policy: Optional[Dict[str, Any]], use_reattachable_execute: bool, session_hooks: Optional[list[SparkSession.Hook]], allow_arrow_batch_chunking: bool, preferred_arrow_chunk_size: Optional[int])` |
| `sql.session.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.session.SparkConversionMixin` | Class | `(...)` |
| `sql.session.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.session.StreamingQueryManager` | Class | `(jsqm: JavaObject)` |
| `sql.session.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `sql.session.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.session.TableValuedFunction` | Class | `(sparkSession: SparkSession)` |
| `sql.session.UDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.session.UDTFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.session.VariantVal` | Class | `(value: bytes, metadata: bytes)` |
| `sql.session.classproperty` | Class | `(...)` |
| `sql.session.default_api_mode` | Function | `() -> str` |
| `sql.session.install_exception_handler` | Function | `() -> None` |
| `sql.session.is_remote_only` | Function | `() -> bool` |
| `sql.session.is_timestamp_ntz_preferred` | Function | `() -> bool` |
| `sql.session.lit` | Function | `(col: Any) -> Column` |
| `sql.session.remote_only` | Function | `(func: Union[Callable, property]) -> Union[Callable, property]` |
| `sql.session.to_str` | Function | `(value: Any) -> Optional[str]` |
| `sql.session.try_remote_session_classmethod` | Function | `(f: FuncT) -> FuncT` |
| `sql.sql_formatter.DataFrame` | Class | `(...)` |
| `sql.sql_formatter.PySparkValueError` | Class | `(...)` |
| `sql.sql_formatter.SQLStringFormatter` | Class | `(session: SparkSession)` |
| `sql.sql_formatter.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.sql_formatter.get_lit_sql_str` | Function | `(val: str) -> str` |
| `sql.streaming.DataStreamReader` | Class | `(spark: SparkSession)` |
| `sql.streaming.DataStreamWriter` | Class | `(df: DataFrame)` |
| `sql.streaming.StatefulProcessor` | Class | `(...)` |
| `sql.streaming.StatefulProcessorHandle` | Class | `(statefulProcessorApiClient: StatefulProcessorApiClient)` |
| `sql.streaming.StreamingQuery` | Class | `(jsq: JavaObject)` |
| `sql.streaming.StreamingQueryException` | Class | `(...)` |
| `sql.streaming.StreamingQueryListener` | Class | `(...)` |
| `sql.streaming.StreamingQueryManager` | Class | `(jsqm: JavaObject)` |
| `sql.streaming.list_state_client.ListStateClient` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, schema: Union[StructType, str])` |
| `sql.streaming.list_state_client.ListStateIterator` | Class | `(list_state_client: ListStateClient, state_name: str)` |
| `sql.streaming.list_state_client.PySparkRuntimeError` | Class | `(...)` |
| `sql.streaming.list_state_client.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `sql.streaming.list_state_client.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.listener.JStreamingQueryListener` | Class | `(pylistener: StreamingQueryListener)` |
| `sql.streaming.listener.QueryIdleEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, timestamp: str)` |
| `sql.streaming.listener.QueryProgressEvent` | Class | `(progress: StreamingQueryProgress)` |
| `sql.streaming.listener.QueryStartedEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, name: Optional[str], timestamp: str, jobTags: Set[str])` |
| `sql.streaming.listener.QueryTerminatedEvent` | Class | `(id: uuid.UUID, runId: uuid.UUID, exception: Optional[str], errorClassOnException: Optional[str])` |
| `sql.streaming.listener.Row` | Class | `(...)` |
| `sql.streaming.listener.SinkProgress` | Class | `(description: str, numOutputRows: int, metrics: Dict[str, str], jprogress: Optional[JavaObject], jdict: Optional[Dict[str, Any]])` |
| `sql.streaming.listener.SourceProgress` | Class | `(description: str, startOffset: str, endOffset: str, latestOffset: str, numInputRows: int, inputRowsPerSecond: float, processedRowsPerSecond: float, metrics: Dict[str, str], jprogress: Optional[JavaObject], jdict: Optional[Dict[str, Any]])` |
| `sql.streaming.listener.StateOperatorProgress` | Class | `(operatorName: str, numRowsTotal: int, numRowsUpdated: int, numRowsRemoved: int, allUpdatesTimeMs: int, allRemovalsTimeMs: int, commitTimeMs: int, memoryUsedBytes: int, numRowsDroppedByWatermark: int, numShufflePartitions: int, numStateStoreInstances: int, customMetrics: Dict[str, int], jprogress: Optional[JavaObject], jdict: Optional[Dict[str, Any]])` |
| `sql.streaming.listener.StreamingQueryListener` | Class | `(...)` |
| `sql.streaming.listener.StreamingQueryProgress` | Class | `(id: uuid.UUID, runId: uuid.UUID, name: Optional[str], timestamp: str, batchId: int, batchDuration: int, durationMs: Dict[str, int], eventTime: Dict[str, str], stateOperators: List[StateOperatorProgress], sources: List[SourceProgress], sink: SinkProgress, numInputRows: int, inputRowsPerSecond: float, processedRowsPerSecond: float, observedMetrics: Dict[str, Row], jprogress: Optional[JavaObject], jdict: Optional[Dict[str, Any]])` |
| `sql.streaming.listener.cloudpickle` | Object | `` |
| `sql.streaming.map_state_client.MapStateClient` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, user_key_schema: Union[StructType, str], value_schema: Union[StructType, str])` |
| `sql.streaming.map_state_client.MapStateIterator` | Class | `(map_state_client: MapStateClient, state_name: str, is_key: bool)` |
| `sql.streaming.map_state_client.MapStateKeyValuePairIterator` | Class | `(map_state_client: MapStateClient, state_name: str)` |
| `sql.streaming.map_state_client.PySparkRuntimeError` | Class | `(...)` |
| `sql.streaming.map_state_client.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `sql.streaming.map_state_client.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.proto.StateMessage_pb2.AppendList` | Class | `(value: collections.abc.Iterable[builtins.bytes] \| None, fetchWithArrow: builtins.bool)` |
| `sql.streaming.proto.StateMessage_pb2.AppendValue` | Class | `(value: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.CLOSED` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.CREATED` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.Clear` | Class | `()` |
| `sql.streaming.proto.StateMessage_pb2.ContainsKey` | Class | `(userKey: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.DATA_PROCESSED` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.DESCRIPTOR` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.DeleteTimer` | Class | `(expiryTimestampMs: builtins.int)` |
| `sql.streaming.proto.StateMessage_pb2.Exists` | Class | `()` |
| `sql.streaming.proto.StateMessage_pb2.ExpiryTimerRequest` | Class | `(iteratorId: builtins.str, expiryTimestampMs: builtins.int)` |
| `sql.streaming.proto.StateMessage_pb2.Get` | Class | `()` |
| `sql.streaming.proto.StateMessage_pb2.GetProcessingTime` | Class | `()` |
| `sql.streaming.proto.StateMessage_pb2.GetValue` | Class | `(userKey: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.GetWatermark` | Class | `()` |
| `sql.streaming.proto.StateMessage_pb2.HandleState` | Class | `(...)` |
| `sql.streaming.proto.StateMessage_pb2.INITIALIZED` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.ImplicitGroupingKeyRequest` | Class | `(setImplicitKey: global___SetImplicitKey \| None, removeImplicitKey: global___RemoveImplicitKey \| None)` |
| `sql.streaming.proto.StateMessage_pb2.Iterator` | Class | `(iteratorId: builtins.str)` |
| `sql.streaming.proto.StateMessage_pb2.KeyAndValuePair` | Class | `(key: builtins.bytes, value: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.Keys` | Class | `(iteratorId: builtins.str)` |
| `sql.streaming.proto.StateMessage_pb2.ListStateCall` | Class | `(stateName: builtins.str, exists: global___Exists \| None, listStateGet: global___ListStateGet \| None, listStatePut: global___ListStatePut \| None, appendValue: global___AppendValue \| None, appendList: global___AppendList \| None, clear: global___Clear \| None)` |
| `sql.streaming.proto.StateMessage_pb2.ListStateGet` | Class | `(iteratorId: builtins.str)` |
| `sql.streaming.proto.StateMessage_pb2.ListStatePut` | Class | `(value: collections.abc.Iterable[builtins.bytes] \| None, fetchWithArrow: builtins.bool)` |
| `sql.streaming.proto.StateMessage_pb2.ListTimers` | Class | `(iteratorId: builtins.str)` |
| `sql.streaming.proto.StateMessage_pb2.MapStateCall` | Class | `(stateName: builtins.str, exists: global___Exists \| None, getValue: global___GetValue \| None, containsKey: global___ContainsKey \| None, updateValue: global___UpdateValue \| None, iterator: global___Iterator \| None, keys: global___Keys \| None, values: global___Values \| None, removeKey: global___RemoveKey \| None, clear: global___Clear \| None)` |
| `sql.streaming.proto.StateMessage_pb2.PRE_INIT` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.ParseStringSchema` | Class | `(schema: builtins.str)` |
| `sql.streaming.proto.StateMessage_pb2.RegisterTimer` | Class | `(expiryTimestampMs: builtins.int)` |
| `sql.streaming.proto.StateMessage_pb2.RemoveImplicitKey` | Class | `()` |
| `sql.streaming.proto.StateMessage_pb2.RemoveKey` | Class | `(userKey: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.SetHandleState` | Class | `(state: global___HandleState.ValueType)` |
| `sql.streaming.proto.StateMessage_pb2.SetImplicitKey` | Class | `(key: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.StateCallCommand` | Class | `(stateName: builtins.str, schema: builtins.str, mapStateValueSchema: builtins.str, ttl: global___TTLConfig \| None)` |
| `sql.streaming.proto.StateMessage_pb2.StateRequest` | Class | `(version: builtins.int, statefulProcessorCall: global___StatefulProcessorCall \| None, stateVariableRequest: global___StateVariableRequest \| None, implicitGroupingKeyRequest: global___ImplicitGroupingKeyRequest \| None, timerRequest: global___TimerRequest \| None, utilsRequest: global___UtilsRequest \| None)` |
| `sql.streaming.proto.StateMessage_pb2.StateResponse` | Class | `(statusCode: builtins.int, errorMessage: builtins.str, value: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.StateResponseWithListGet` | Class | `(statusCode: builtins.int, errorMessage: builtins.str, value: collections.abc.Iterable[builtins.bytes] \| None, requireNextFetch: builtins.bool)` |
| `sql.streaming.proto.StateMessage_pb2.StateResponseWithLongTypeVal` | Class | `(statusCode: builtins.int, errorMessage: builtins.str, value: builtins.int)` |
| `sql.streaming.proto.StateMessage_pb2.StateResponseWithMapIterator` | Class | `(statusCode: builtins.int, errorMessage: builtins.str, kvPair: collections.abc.Iterable[global___KeyAndValuePair] \| None, requireNextFetch: builtins.bool)` |
| `sql.streaming.proto.StateMessage_pb2.StateResponseWithMapKeysOrValues` | Class | `(statusCode: builtins.int, errorMessage: builtins.str, value: collections.abc.Iterable[builtins.bytes] \| None, requireNextFetch: builtins.bool)` |
| `sql.streaming.proto.StateMessage_pb2.StateResponseWithStringTypeVal` | Class | `(statusCode: builtins.int, errorMessage: builtins.str, value: builtins.str)` |
| `sql.streaming.proto.StateMessage_pb2.StateResponseWithTimer` | Class | `(statusCode: builtins.int, errorMessage: builtins.str, timer: collections.abc.Iterable[global___TimerInfo] \| None, requireNextFetch: builtins.bool)` |
| `sql.streaming.proto.StateMessage_pb2.StateVariableRequest` | Class | `(valueStateCall: global___ValueStateCall \| None, listStateCall: global___ListStateCall \| None, mapStateCall: global___MapStateCall \| None)` |
| `sql.streaming.proto.StateMessage_pb2.StatefulProcessorCall` | Class | `(setHandleState: global___SetHandleState \| None, getValueState: global___StateCallCommand \| None, getListState: global___StateCallCommand \| None, getMapState: global___StateCallCommand \| None, timerStateCall: global___TimerStateCallCommand \| None, deleteIfExists: global___StateCallCommand \| None)` |
| `sql.streaming.proto.StateMessage_pb2.TIMER_PROCESSED` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.TTLConfig` | Class | `(durationMs: builtins.int)` |
| `sql.streaming.proto.StateMessage_pb2.TimerInfo` | Class | `(key: builtins.bytes \| None, timestampMs: builtins.int)` |
| `sql.streaming.proto.StateMessage_pb2.TimerRequest` | Class | `(timerValueRequest: global___TimerValueRequest \| None, expiryTimerRequest: global___ExpiryTimerRequest \| None)` |
| `sql.streaming.proto.StateMessage_pb2.TimerStateCallCommand` | Class | `(register: global___RegisterTimer \| None, delete: global___DeleteTimer \| None, list: global___ListTimers \| None)` |
| `sql.streaming.proto.StateMessage_pb2.TimerValueRequest` | Class | `(getProcessingTimer: global___GetProcessingTime \| None, getWatermark: global___GetWatermark \| None)` |
| `sql.streaming.proto.StateMessage_pb2.UpdateValue` | Class | `(userKey: builtins.bytes, value: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.UtilsRequest` | Class | `(parseStringSchema: global___ParseStringSchema \| None)` |
| `sql.streaming.proto.StateMessage_pb2.ValueStateCall` | Class | `(stateName: builtins.str, exists: global___Exists \| None, get: global___Get \| None, valueStateUpdate: global___ValueStateUpdate \| None, clear: global___Clear \| None)` |
| `sql.streaming.proto.StateMessage_pb2.ValueStateUpdate` | Class | `(value: builtins.bytes)` |
| `sql.streaming.proto.StateMessage_pb2.Values` | Class | `(iteratorId: builtins.str)` |
| `sql.streaming.proto.StateMessage_pb2.global___AppendList` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___AppendValue` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___Clear` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ContainsKey` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___DeleteTimer` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___Exists` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ExpiryTimerRequest` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___Get` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___GetProcessingTime` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___GetValue` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___GetWatermark` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___HandleState` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ImplicitGroupingKeyRequest` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___Iterator` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___KeyAndValuePair` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___Keys` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ListStateCall` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ListStateGet` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ListStatePut` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ListTimers` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___MapStateCall` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ParseStringSchema` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___RegisterTimer` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___RemoveImplicitKey` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___RemoveKey` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___SetHandleState` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___SetImplicitKey` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateCallCommand` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateRequest` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateResponse` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateResponseWithListGet` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateResponseWithLongTypeVal` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateResponseWithMapIterator` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateResponseWithMapKeysOrValues` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateResponseWithStringTypeVal` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateResponseWithTimer` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StateVariableRequest` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___StatefulProcessorCall` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___TTLConfig` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___TimerInfo` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___TimerRequest` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___TimerStateCallCommand` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___TimerValueRequest` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___UpdateValue` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___UtilsRequest` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ValueStateCall` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___ValueStateUpdate` | Object | `` |
| `sql.streaming.proto.StateMessage_pb2.global___Values` | Object | `` |
| `sql.streaming.python_streaming_source_runner.ArrowStreamSerializer` | Class | `(...)` |
| `sql.streaming.python_streaming_source_runner.COMMIT_FUNC_ID` | Object | `` |
| `sql.streaming.python_streaming_source_runner.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.streaming.python_streaming_source_runner.DataSourceStreamReader` | Class | `(...)` |
| `sql.streaming.python_streaming_source_runner.EMPTY_PYARROW_RECORD_BATCHES` | Object | `` |
| `sql.streaming.python_streaming_source_runner.INITIAL_OFFSET_FUNC_ID` | Object | `` |
| `sql.streaming.python_streaming_source_runner.IllegalArgumentException` | Class | `(...)` |
| `sql.streaming.python_streaming_source_runner.LATEST_OFFSET_FUNC_ID` | Object | `` |
| `sql.streaming.python_streaming_source_runner.NON_EMPTY_PYARROW_RECORD_BATCHES` | Object | `` |
| `sql.streaming.python_streaming_source_runner.PARTITIONS_FUNC_ID` | Object | `` |
| `sql.streaming.python_streaming_source_runner.PREFETCHED_RECORDS_NOT_FOUND` | Object | `` |
| `sql.streaming.python_streaming_source_runner.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.streaming.python_streaming_source_runner.SpecialLengths` | Class | `(...)` |
| `sql.streaming.python_streaming_source_runner.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.python_streaming_source_runner.auth_secret` | Object | `` |
| `sql.streaming.python_streaming_source_runner.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.commit_func` | Function | `(reader: DataSourceStreamReader, infile: IO, outfile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.conn_info` | Object | `` |
| `sql.streaming.python_streaming_source_runner.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.streaming.python_streaming_source_runner.initial_offset_func` | Function | `(reader: DataSourceStreamReader, outfile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.latest_offset_func` | Function | `(reader: DataSourceStreamReader, outfile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.streaming.python_streaming_source_runner.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.partitions_func` | Function | `(reader: DataSourceStreamReader, data_source: DataSource, schema: StructType, max_arrow_batch_size: int, infile: IO, outfile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.pickleSer` | Object | `` |
| `sql.streaming.python_streaming_source_runner.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `sql.streaming.python_streaming_source_runner.read_int` | Function | `(stream)` |
| `sql.streaming.python_streaming_source_runner.records_to_arrow_batches` | Function | `(output_iter: Union[Iterator[Tuple], Iterator[pa.RecordBatch]], max_arrow_batch_size: int, return_type: StructType, data_source: DataSource) -> Iterable[pa.RecordBatch]` |
| `sql.streaming.python_streaming_source_runner.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.send_batch_func` | Function | `(rows: Iterator[Tuple], outfile: IO, schema: StructType, max_arrow_batch_size: int, data_source: DataSource) -> None` |
| `sql.streaming.python_streaming_source_runner.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.streaming.python_streaming_source_runner.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.streaming.python_streaming_source_runner.utf8_deserializer` | Object | `` |
| `sql.streaming.python_streaming_source_runner.write_int` | Function | `(value, stream)` |
| `sql.streaming.python_streaming_source_runner.write_with_length` | Function | `(obj, stream)` |
| `sql.streaming.query.CapturedStreamingQueryException` | Class | `(...)` |
| `sql.streaming.query.PySparkValueError` | Class | `(...)` |
| `sql.streaming.query.StreamingQuery` | Class | `(jsq: JavaObject)` |
| `sql.streaming.query.StreamingQueryException` | Class | `(...)` |
| `sql.streaming.query.StreamingQueryListener` | Class | `(...)` |
| `sql.streaming.query.StreamingQueryManager` | Class | `(jsqm: JavaObject)` |
| `sql.streaming.query.StreamingQueryProgress` | Class | `(id: uuid.UUID, runId: uuid.UUID, name: Optional[str], timestamp: str, batchId: int, batchDuration: int, durationMs: Dict[str, int], eventTime: Dict[str, str], stateOperators: List[StateOperatorProgress], sources: List[SourceProgress], sink: SinkProgress, numInputRows: int, inputRowsPerSecond: float, processedRowsPerSecond: float, observedMetrics: Dict[str, Row], jprogress: Optional[JavaObject], jdict: Optional[Dict[str, Any]])` |
| `sql.streaming.readwriter.DataFrame` | Class | `(...)` |
| `sql.streaming.readwriter.DataStreamReader` | Class | `(spark: SparkSession)` |
| `sql.streaming.readwriter.DataStreamWriter` | Class | `(df: DataFrame)` |
| `sql.streaming.readwriter.ForeachBatchFunction` | Class | `(session: SparkSession, func: Callable[[DataFrame, int], None])` |
| `sql.streaming.readwriter.OptionUtils` | Class | `(...)` |
| `sql.streaming.readwriter.OptionalPrimitiveType` | Object | `` |
| `sql.streaming.readwriter.PySparkAttributeError` | Class | `(...)` |
| `sql.streaming.readwriter.PySparkRuntimeError` | Class | `(...)` |
| `sql.streaming.readwriter.PySparkTypeError` | Class | `(...)` |
| `sql.streaming.readwriter.PySparkValueError` | Class | `(...)` |
| `sql.streaming.readwriter.Row` | Class | `(...)` |
| `sql.streaming.readwriter.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.streaming.readwriter.StreamingQuery` | Class | `(jsq: JavaObject)` |
| `sql.streaming.readwriter.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.readwriter.SupportsProcess` | Class | `(...)` |
| `sql.streaming.readwriter.to_str` | Function | `(value: Any) -> Optional[str]` |
| `sql.streaming.state.GroupState` | Class | `(optionalValue: Row, batchProcessingTimeMs: int, eventTimeWatermarkMs: int, timeoutConf: str, hasTimedOut: bool, watermarkPresent: bool, defined: bool, updated: bool, removed: bool, timeoutTimestamp: int, keyAsUnsafe: bytes, valueSchema: StructType)` |
| `sql.streaming.state.GroupStateTimeout` | Class | `(...)` |
| `sql.streaming.state.PySparkRuntimeError` | Class | `(...)` |
| `sql.streaming.state.PySparkTypeError` | Class | `(...)` |
| `sql.streaming.state.PySparkValueError` | Class | `(...)` |
| `sql.streaming.state.Row` | Class | `(...)` |
| `sql.streaming.state.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.state.TimestampType` | Class | `(...)` |
| `sql.streaming.stateful_processor.ExpiredTimerInfo` | Class | `(expiryTimeInMs: int)` |
| `sql.streaming.stateful_processor.ListState` | Class | `(listStateClient: ListStateClient, stateName: str)` |
| `sql.streaming.stateful_processor.ListStateClient` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, schema: Union[StructType, str])` |
| `sql.streaming.stateful_processor.ListStateIterator` | Class | `(list_state_client: ListStateClient, state_name: str)` |
| `sql.streaming.stateful_processor.ListTimerIterator` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient)` |
| `sql.streaming.stateful_processor.MapState` | Class | `(mapStateClient: MapStateClient, stateName: str)` |
| `sql.streaming.stateful_processor.MapStateClient` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, user_key_schema: Union[StructType, str], value_schema: Union[StructType, str])` |
| `sql.streaming.stateful_processor.MapStateIterator` | Class | `(map_state_client: MapStateClient, state_name: str, is_key: bool)` |
| `sql.streaming.stateful_processor.MapStateKeyValuePairIterator` | Class | `(map_state_client: MapStateClient, state_name: str)` |
| `sql.streaming.stateful_processor.PandasDataFrameLike` | Object | `` |
| `sql.streaming.stateful_processor.Row` | Class | `(...)` |
| `sql.streaming.stateful_processor.StatefulProcessor` | Class | `(...)` |
| `sql.streaming.stateful_processor.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `sql.streaming.stateful_processor.StatefulProcessorHandle` | Class | `(statefulProcessorApiClient: StatefulProcessorApiClient)` |
| `sql.streaming.stateful_processor.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.stateful_processor.TimerValues` | Class | `(currentProcessingTimeInMs: int, currentWatermarkInMs: int)` |
| `sql.streaming.stateful_processor.ValueState` | Class | `(valueStateClient: ValueStateClient, stateName: str)` |
| `sql.streaming.stateful_processor.ValueStateClient` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, schema: Union[StructType, str])` |
| `sql.streaming.stateful_processor_api_client.ArrowStreamSerializer` | Class | `(...)` |
| `sql.streaming.stateful_processor_api_client.CPickleSerializer` | Object | `` |
| `sql.streaming.stateful_processor_api_client.ExpiredTimerIterator` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, expiry_timestamp: int)` |
| `sql.streaming.stateful_processor_api_client.ListTimerIterator` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient)` |
| `sql.streaming.stateful_processor_api_client.PySparkRuntimeError` | Class | `(...)` |
| `sql.streaming.stateful_processor_api_client.Row` | Class | `(...)` |
| `sql.streaming.stateful_processor_api_client.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `sql.streaming.stateful_processor_api_client.StatefulProcessorHandleState` | Class | `(...)` |
| `sql.streaming.stateful_processor_api_client.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.stateful_processor_api_client.UTF8Deserializer` | Class | `(use_unicode)` |
| `sql.streaming.stateful_processor_api_client.convert_pandas_using_numpy_type` | Function | `(df: PandasDataFrameLike, schema: StructType) -> PandasDataFrameLike` |
| `sql.streaming.stateful_processor_api_client.read_int` | Function | `(stream)` |
| `sql.streaming.stateful_processor_api_client.write_int` | Function | `(value, stream)` |
| `sql.streaming.stateful_processor_util.ExpiredTimerInfo` | Class | `(expiryTimeInMs: int)` |
| `sql.streaming.stateful_processor_util.ExpiredTimerIterator` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, expiry_timestamp: int)` |
| `sql.streaming.stateful_processor_util.PandasDataFrameLike` | Object | `` |
| `sql.streaming.stateful_processor_util.Row` | Class | `(...)` |
| `sql.streaming.stateful_processor_util.StatefulProcessor` | Class | `(...)` |
| `sql.streaming.stateful_processor_util.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `sql.streaming.stateful_processor_util.StatefulProcessorHandle` | Class | `(statefulProcessorApiClient: StatefulProcessorApiClient)` |
| `sql.streaming.stateful_processor_util.StatefulProcessorHandleState` | Class | `(...)` |
| `sql.streaming.stateful_processor_util.TimerValues` | Class | `(currentProcessingTimeInMs: int, currentWatermarkInMs: int)` |
| `sql.streaming.stateful_processor_util.TransformWithStateInPandasFuncMode` | Class | `(...)` |
| `sql.streaming.stateful_processor_util.TransformWithStateInPandasUdfUtils` | Class | `(stateful_processor: StatefulProcessor, time_mode: str)` |
| `sql.streaming.transform_with_state_driver_worker.CPickleSerializer` | Object | `` |
| `sql.streaming.transform_with_state_driver_worker.PandasDataFrameLike` | Object | `` |
| `sql.streaming.transform_with_state_driver_worker.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `sql.streaming.transform_with_state_driver_worker.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.transform_with_state_driver_worker.TransformWithStateInPandasFuncMode` | Class | `(...)` |
| `sql.streaming.transform_with_state_driver_worker.UTF8Deserializer` | Class | `(use_unicode)` |
| `sql.streaming.transform_with_state_driver_worker.auth_secret` | Object | `` |
| `sql.streaming.transform_with_state_driver_worker.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.streaming.transform_with_state_driver_worker.conn_info` | Object | `` |
| `sql.streaming.transform_with_state_driver_worker.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.streaming.transform_with_state_driver_worker.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.streaming.transform_with_state_driver_worker.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.streaming.transform_with_state_driver_worker.pickle_ser` | Object | `` |
| `sql.streaming.transform_with_state_driver_worker.read_int` | Function | `(stream)` |
| `sql.streaming.transform_with_state_driver_worker.utf8_deserializer` | Object | `` |
| `sql.streaming.transform_with_state_driver_worker.worker` | Object | `` |
| `sql.streaming.transform_with_state_driver_worker.write_int` | Function | `(value, stream)` |
| `sql.streaming.value_state_client.PySparkRuntimeError` | Class | `(...)` |
| `sql.streaming.value_state_client.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `sql.streaming.value_state_client.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.streaming.value_state_client.ValueStateClient` | Class | `(stateful_processor_api_client: StatefulProcessorApiClient, schema: Union[StructType, str])` |
| `sql.table_arg.ColumnOrName` | Object | `` |
| `sql.table_arg.TableArg` | Class | `(...)` |
| `sql.table_arg.TableValuedFunctionArgument` | Class | `(...)` |
| `sql.table_arg.dispatch_table_arg_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.tvf.Column` | Class | `(...)` |
| `sql.tvf.DataFrame` | Class | `(...)` |
| `sql.tvf.PySparkValueError` | Class | `(...)` |
| `sql.tvf.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.tvf.TableValuedFunction` | Class | `(sparkSession: SparkSession)` |
| `sql.tvf_argument.TableValuedFunctionArgument` | Class | `(...)` |
| `sql.types.AnsiIntervalType` | Class | `(...)` |
| `sql.types.AnyTimeType` | Class | `(...)` |
| `sql.types.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `sql.types.AtomicType` | Class | `(...)` |
| `sql.types.BinaryType` | Class | `(...)` |
| `sql.types.BooleanType` | Class | `(...)` |
| `sql.types.ByteType` | Class | `(...)` |
| `sql.types.CalendarIntervalType` | Class | `(...)` |
| `sql.types.CharType` | Class | `(length: int)` |
| `sql.types.CloudPickleSerializer` | Class | `(...)` |
| `sql.types.DataType` | Class | `(...)` |
| `sql.types.DataTypeSingleton` | Class | `(...)` |
| `sql.types.DateConverter` | Class | `(...)` |
| `sql.types.DateType` | Class | `(...)` |
| `sql.types.DatetimeConverter` | Class | `(...)` |
| `sql.types.DatetimeNTZConverter` | Class | `(...)` |
| `sql.types.DatetimeType` | Class | `(...)` |
| `sql.types.DayTimeIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `sql.types.DayTimeIntervalTypeConverter` | Class | `(...)` |
| `sql.types.DecimalType` | Class | `(precision: int, scale: int)` |
| `sql.types.DoubleType` | Class | `(...)` |
| `sql.types.FloatType` | Class | `(...)` |
| `sql.types.FractionalType` | Class | `(...)` |
| `sql.types.Geography` | Class | `(wkb: bytes, srid: int)` |
| `sql.types.GeographyType` | Class | `(srid: int \| str)` |
| `sql.types.Geometry` | Class | `(wkb: bytes, srid: int)` |
| `sql.types.GeometryType` | Class | `(srid: int \| str)` |
| `sql.types.IllegalArgumentException` | Class | `(...)` |
| `sql.types.IntegerType` | Class | `(...)` |
| `sql.types.IntegralType` | Class | `(...)` |
| `sql.types.JVM_INT_MAX` | Object | `` |
| `sql.types.LongType` | Class | `(...)` |
| `sql.types.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `sql.types.NullType` | Class | `(...)` |
| `sql.types.NumericType` | Class | `(...)` |
| `sql.types.NumpyArrayConverter` | Class | `(...)` |
| `sql.types.NumpyScalarConverter` | Class | `(...)` |
| `sql.types.PySparkAttributeError` | Class | `(...)` |
| `sql.types.PySparkIndexError` | Class | `(...)` |
| `sql.types.PySparkKeyError` | Class | `(...)` |
| `sql.types.PySparkNotImplementedError` | Class | `(...)` |
| `sql.types.PySparkRuntimeError` | Class | `(...)` |
| `sql.types.PySparkTypeError` | Class | `(...)` |
| `sql.types.PySparkValueError` | Class | `(...)` |
| `sql.types.Row` | Class | `(...)` |
| `sql.types.ShortType` | Class | `(...)` |
| `sql.types.SpatialType` | Class | `(...)` |
| `sql.types.StringConcat` | Class | `(maxLength: int)` |
| `sql.types.StringType` | Class | `(collation: str)` |
| `sql.types.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `sql.types.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.types.T` | Object | `` |
| `sql.types.TimeConverter` | Class | `(...)` |
| `sql.types.TimeType` | Class | `(precision: int)` |
| `sql.types.TimestampNTZType` | Class | `(...)` |
| `sql.types.TimestampType` | Class | `(...)` |
| `sql.types.U` | Object | `` |
| `sql.types.UserDefinedType` | Class | `(...)` |
| `sql.types.VarcharType` | Class | `(length: int)` |
| `sql.types.VariantType` | Class | `(...)` |
| `sql.types.VariantUtils` | Class | `(...)` |
| `sql.types.VariantVal` | Class | `(value: bytes, metadata: bytes)` |
| `sql.types.YearMonthIntervalType` | Class | `(startField: Optional[int], endField: Optional[int])` |
| `sql.types.dt` | Object | `` |
| `sql.types.escape_meta_characters` | Function | `(s: str) -> str` |
| `sql.types.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.types.is_remote_only` | Function | `() -> bool` |
| `sql.types.size` | Object | `` |
| `sql.udf.Column` | Class | `(...)` |
| `sql.udf.ColumnOrName` | Object | `` |
| `sql.udf.DataType` | Class | `(...)` |
| `sql.udf.DataTypeOrString` | Object | `` |
| `sql.udf.PySparkNotImplementedError` | Class | `(...)` |
| `sql.udf.PySparkRuntimeError` | Class | `(...)` |
| `sql.udf.PySparkTypeError` | Class | `(...)` |
| `sql.udf.PythonEvalType` | Class | `(...)` |
| `sql.udf.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.udf.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.udf.StringType` | Class | `(collation: str)` |
| `sql.udf.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.udf.UDFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.udf.UserDefinedFunction` | Class | `(func: Callable[..., Any], returnType: DataTypeOrString, name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.udf.UserDefinedFunctionLike` | Class | `(...)` |
| `sql.udf.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.udf.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.udf.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.udf.to_arrow_type` | Function | `(dt: DataType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.DataType` |
| `sql.udtf.AnalyzeArgument` | Class | `(dataType: DataType, value: Optional[Any], isTable: bool, isConstantExpression: bool)` |
| `sql.udtf.AnalyzeResult` | Class | `(schema: StructType, withSinglePartition: bool, partitionBy: Sequence[PartitioningColumn], orderBy: Sequence[OrderingColumn], select: Sequence[SelectedColumn])` |
| `sql.udtf.DataFrame` | Class | `(...)` |
| `sql.udtf.DataType` | Class | `(...)` |
| `sql.udtf.OrderingColumn` | Class | `(name: str, ascending: bool, overrideNullsFirst: Optional[bool])` |
| `sql.udtf.PartitioningColumn` | Class | `(name: str)` |
| `sql.udtf.PySparkAttributeError` | Class | `(...)` |
| `sql.udtf.PySparkImportError` | Class | `(...)` |
| `sql.udtf.PySparkPicklingError` | Class | `(...)` |
| `sql.udtf.PySparkTypeError` | Class | `(...)` |
| `sql.udtf.PythonEvalType` | Class | `(...)` |
| `sql.udtf.SelectedColumn` | Class | `(name: str, alias: str)` |
| `sql.udtf.SkipRestOfInputTableException` | Class | `(...)` |
| `sql.udtf.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.udtf.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.udtf.TVFArgumentOrName` | Object | `` |
| `sql.udtf.UDTFRegistration` | Class | `(sparkSession: SparkSession)` |
| `sql.udtf.UserDefinedTableFunction` | Class | `(func: Type, returnType: Optional[Union[StructType, str]], name: Optional[str], evalType: int, deterministic: bool)` |
| `sql.udtf.require_minimum_pandas_version` | Function | `() -> None` |
| `sql.udtf.require_minimum_pyarrow_version` | Function | `() -> None` |
| `sql.utils.AnalysisException` | Class | `(...)` |
| `sql.utils.CapturedException` | Class | `(desc: Optional[str], stackTrace: Optional[str], cause: Optional[Py4JJavaError], origin: Optional[Py4JJavaError])` |
| `sql.utils.DataFrame` | Class | `(...)` |
| `sql.utils.ForeachBatchFunction` | Class | `(session: SparkSession, func: Callable[[DataFrame, int], None])` |
| `sql.utils.FuncT` | Object | `` |
| `sql.utils.IllegalArgumentException` | Class | `(...)` |
| `sql.utils.IndexOpsLike` | Object | `` |
| `sql.utils.JVM_INT_MAX` | Object | `` |
| `sql.utils.NumpyHelper` | Class | `(...)` |
| `sql.utils.ParseException` | Class | `(...)` |
| `sql.utils.PySparkImportError` | Class | `(...)` |
| `sql.utils.PySparkNotImplementedError` | Class | `(...)` |
| `sql.utils.PySparkRuntimeError` | Class | `(...)` |
| `sql.utils.PythonException` | Class | `(...)` |
| `sql.utils.QueryExecutionException` | Class | `(...)` |
| `sql.utils.SeriesOrIndex` | Object | `` |
| `sql.utils.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `sql.utils.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `sql.utils.SparkUpgradeException` | Class | `(...)` |
| `sql.utils.StreamingQueryException` | Class | `(...)` |
| `sql.utils.StringConcat` | Class | `(maxLength: int)` |
| `sql.utils.UnknownException` | Class | `(...)` |
| `sql.utils.dispatch_col_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.dispatch_df_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.dispatch_table_arg_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.dispatch_window_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.enum_to_value` | Function | `(value: Any) -> Any` |
| `sql.utils.escape_meta_characters` | Function | `(s: str) -> str` |
| `sql.utils.get_active_spark_context` | Function | `() -> SparkContext` |
| `sql.utils.get_lit_sql_str` | Function | `(val: str) -> str` |
| `sql.utils.is_remote` | Function | `() -> bool` |
| `sql.utils.is_remote_only` | Function | `() -> bool` |
| `sql.utils.is_timestamp_ntz_preferred` | Function | `() -> bool` |
| `sql.utils.pyspark_column_op` | Function | `(func_name: str, left: IndexOpsLike, right: Any, fillna: Any) -> Union[SeriesOrIndex, None]` |
| `sql.utils.remote_only` | Function | `(func: Union[Callable, property]) -> Union[Callable, property]` |
| `sql.utils.require_minimum_plotly_version` | Function | `() -> None` |
| `sql.utils.require_test_compiled` | Function | `() -> None` |
| `sql.utils.to_java_array` | Function | `(gateway: JavaGateway, jtype: JavaClass, arr: Sequence[Any]) -> JavaArray` |
| `sql.utils.to_scala_map` | Function | `(jvm: JVMView, dic: Dict) -> JavaObject` |
| `sql.utils.to_str` | Function | `(value: Any) -> Optional[str]` |
| `sql.utils.try_partitioning_remote_functions` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.try_remote_avro_functions` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.try_remote_functions` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.try_remote_protobuf_functions` | Function | `(f: FuncT) -> FuncT` |
| `sql.utils.try_remote_session_classmethod` | Function | `(f: FuncT) -> FuncT` |
| `sql.variant_utils.FieldEntry` | Class | `(...)` |
| `sql.variant_utils.PySparkValueError` | Class | `(...)` |
| `sql.variant_utils.VariantBuilder` | Class | `(size_limit: int)` |
| `sql.variant_utils.VariantUtils` | Class | `(...)` |
| `sql.window.ColumnOrName` | Object | `` |
| `sql.window.JVM_LONG_MAX` | Object | `` |
| `sql.window.JVM_LONG_MIN` | Object | `` |
| `sql.window.Window` | Class | `(...)` |
| `sql.window.WindowSpec` | Class | `(...)` |
| `sql.window.dispatch_window_method` | Function | `(f: FuncT) -> FuncT` |
| `sql.worker.analyze_udtf.AnalyzeArgument` | Class | `(dataType: DataType, value: Optional[Any], isTable: bool, isConstantExpression: bool)` |
| `sql.worker.analyze_udtf.AnalyzeResult` | Class | `(schema: StructType, withSinglePartition: bool, partitionBy: Sequence[PartitioningColumn], orderBy: Sequence[OrderingColumn], select: Sequence[SelectedColumn])` |
| `sql.worker.analyze_udtf.OrderingColumn` | Class | `(name: str, ascending: bool, overrideNullsFirst: Optional[bool])` |
| `sql.worker.analyze_udtf.PartitioningColumn` | Class | `(name: str)` |
| `sql.worker.analyze_udtf.PySparkRuntimeError` | Class | `(...)` |
| `sql.worker.analyze_udtf.PySparkValueError` | Class | `(...)` |
| `sql.worker.analyze_udtf.SelectedColumn` | Class | `(name: str, alias: str)` |
| `sql.worker.analyze_udtf.SpecialLengths` | Class | `(...)` |
| `sql.worker.analyze_udtf.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.worker.analyze_udtf.auth_secret` | Object | `` |
| `sql.worker.analyze_udtf.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `sql.worker.analyze_udtf.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.analyze_udtf.conn_info` | Object | `` |
| `sql.worker.analyze_udtf.default_context_provider` | Function | `() -> dict[str, str]` |
| `sql.worker.analyze_udtf.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.analyze_udtf.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.analyze_udtf.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.analyze_udtf.pickleSer` | Object | `` |
| `sql.worker.analyze_udtf.read_arguments` | Function | `(infile: IO) -> Tuple[List[AnalyzeArgument], Dict[str, AnalyzeArgument]]` |
| `sql.worker.analyze_udtf.read_bool` | Function | `(stream)` |
| `sql.worker.analyze_udtf.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `sql.worker.analyze_udtf.read_int` | Function | `(stream)` |
| `sql.worker.analyze_udtf.read_udtf` | Function | `(infile: IO) -> type` |
| `sql.worker.analyze_udtf.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.analyze_udtf.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.analyze_udtf.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.analyze_udtf.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.analyze_udtf.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.analyze_udtf.utf8_deserializer` | Object | `` |
| `sql.worker.analyze_udtf.with_faulthandler` | Object | `` |
| `sql.worker.analyze_udtf.write_int` | Function | `(value, stream)` |
| `sql.worker.analyze_udtf.write_with_length` | Function | `(obj, stream)` |
| `sql.worker.commit_data_source_write.DataSourceWriter` | Class | `(...)` |
| `sql.worker.commit_data_source_write.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.worker.commit_data_source_write.SpecialLengths` | Class | `(...)` |
| `sql.worker.commit_data_source_write.WriterCommitMessage` | Class | `(...)` |
| `sql.worker.commit_data_source_write.auth_secret` | Object | `` |
| `sql.worker.commit_data_source_write.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `sql.worker.commit_data_source_write.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.commit_data_source_write.conn_info` | Object | `` |
| `sql.worker.commit_data_source_write.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.commit_data_source_write.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.commit_data_source_write.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.commit_data_source_write.pickleSer` | Object | `` |
| `sql.worker.commit_data_source_write.read_bool` | Function | `(stream)` |
| `sql.worker.commit_data_source_write.read_int` | Function | `(stream)` |
| `sql.worker.commit_data_source_write.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.commit_data_source_write.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.commit_data_source_write.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.commit_data_source_write.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.commit_data_source_write.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.commit_data_source_write.with_faulthandler` | Object | `` |
| `sql.worker.commit_data_source_write.write_int` | Function | `(value, stream)` |
| `sql.worker.create_data_source.CaseInsensitiveDict` | Class | `(args: Any, kwargs: Any)` |
| `sql.worker.create_data_source.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.worker.create_data_source.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.worker.create_data_source.PySparkTypeError` | Class | `(...)` |
| `sql.worker.create_data_source.SpecialLengths` | Class | `(...)` |
| `sql.worker.create_data_source.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.worker.create_data_source.auth_secret` | Object | `` |
| `sql.worker.create_data_source.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `sql.worker.create_data_source.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.create_data_source.conn_info` | Object | `` |
| `sql.worker.create_data_source.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.create_data_source.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.create_data_source.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.create_data_source.pickleSer` | Object | `` |
| `sql.worker.create_data_source.read_bool` | Function | `(stream)` |
| `sql.worker.create_data_source.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `sql.worker.create_data_source.read_int` | Function | `(stream)` |
| `sql.worker.create_data_source.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.create_data_source.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.create_data_source.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.create_data_source.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.create_data_source.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.create_data_source.utf8_deserializer` | Object | `` |
| `sql.worker.create_data_source.with_faulthandler` | Object | `` |
| `sql.worker.create_data_source.write_int` | Function | `(value, stream)` |
| `sql.worker.create_data_source.write_with_length` | Function | `(obj, stream)` |
| `sql.worker.data_source_pushdown_filters.BinaryFilter` | Object | `` |
| `sql.worker.data_source_pushdown_filters.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.worker.data_source_pushdown_filters.DataSourceReader` | Class | `(...)` |
| `sql.worker.data_source_pushdown_filters.EqualNullSafe` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.worker.data_source_pushdown_filters.EqualTo` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.worker.data_source_pushdown_filters.Filter` | Class | `()` |
| `sql.worker.data_source_pushdown_filters.FilterRef` | Class | `(filter: Filter)` |
| `sql.worker.data_source_pushdown_filters.GreaterThan` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.worker.data_source_pushdown_filters.GreaterThanOrEqual` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.worker.data_source_pushdown_filters.In` | Class | `(attribute: ColumnPath, value: Tuple[Any, ...])` |
| `sql.worker.data_source_pushdown_filters.IsNotNull` | Class | `(attribute: ColumnPath)` |
| `sql.worker.data_source_pushdown_filters.IsNull` | Class | `(attribute: ColumnPath)` |
| `sql.worker.data_source_pushdown_filters.LessThan` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.worker.data_source_pushdown_filters.LessThanOrEqual` | Class | `(attribute: ColumnPath, value: Any)` |
| `sql.worker.data_source_pushdown_filters.Not` | Class | `(child: Filter)` |
| `sql.worker.data_source_pushdown_filters.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.worker.data_source_pushdown_filters.PySparkNotImplementedError` | Class | `(...)` |
| `sql.worker.data_source_pushdown_filters.PySparkValueError` | Class | `(...)` |
| `sql.worker.data_source_pushdown_filters.SpecialLengths` | Class | `(...)` |
| `sql.worker.data_source_pushdown_filters.StringContains` | Class | `(attribute: ColumnPath, value: str)` |
| `sql.worker.data_source_pushdown_filters.StringEndsWith` | Class | `(attribute: ColumnPath, value: str)` |
| `sql.worker.data_source_pushdown_filters.StringStartsWith` | Class | `(attribute: ColumnPath, value: str)` |
| `sql.worker.data_source_pushdown_filters.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.worker.data_source_pushdown_filters.UTF8Deserializer` | Class | `(use_unicode)` |
| `sql.worker.data_source_pushdown_filters.UnaryFilter` | Object | `` |
| `sql.worker.data_source_pushdown_filters.VariantVal` | Class | `(value: bytes, metadata: bytes)` |
| `sql.worker.data_source_pushdown_filters.auth_secret` | Object | `` |
| `sql.worker.data_source_pushdown_filters.binary_filters` | Object | `` |
| `sql.worker.data_source_pushdown_filters.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `sql.worker.data_source_pushdown_filters.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.data_source_pushdown_filters.conn_info` | Object | `` |
| `sql.worker.data_source_pushdown_filters.deserializeFilter` | Function | `(jsonDict: dict) -> Filter` |
| `sql.worker.data_source_pushdown_filters.deserializeVariant` | Function | `(variantDict: dict) -> VariantVal` |
| `sql.worker.data_source_pushdown_filters.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.data_source_pushdown_filters.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.data_source_pushdown_filters.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.data_source_pushdown_filters.pickleSer` | Object | `` |
| `sql.worker.data_source_pushdown_filters.read_bool` | Function | `(stream)` |
| `sql.worker.data_source_pushdown_filters.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `sql.worker.data_source_pushdown_filters.read_int` | Function | `(stream)` |
| `sql.worker.data_source_pushdown_filters.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.data_source_pushdown_filters.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.data_source_pushdown_filters.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.data_source_pushdown_filters.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.data_source_pushdown_filters.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.data_source_pushdown_filters.unary_filters` | Object | `` |
| `sql.worker.data_source_pushdown_filters.utf8_deserializer` | Object | `` |
| `sql.worker.data_source_pushdown_filters.with_faulthandler` | Object | `` |
| `sql.worker.data_source_pushdown_filters.write_int` | Function | `(value, stream)` |
| `sql.worker.data_source_pushdown_filters.write_read_func_and_partitions` | Function | `(outfile: IO, reader: Union[DataSourceReader, DataSourceStreamReader], data_source: DataSource, schema: StructType, max_arrow_batch_size: int, binary_as_bytes: bool) -> None` |
| `sql.worker.lookup_data_sources.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.worker.lookup_data_sources.SpecialLengths` | Class | `(...)` |
| `sql.worker.lookup_data_sources.auth_secret` | Object | `` |
| `sql.worker.lookup_data_sources.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.lookup_data_sources.conn_info` | Object | `` |
| `sql.worker.lookup_data_sources.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.lookup_data_sources.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.lookup_data_sources.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.lookup_data_sources.pickleSer` | Object | `` |
| `sql.worker.lookup_data_sources.read_int` | Function | `(stream)` |
| `sql.worker.lookup_data_sources.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.lookup_data_sources.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.lookup_data_sources.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.lookup_data_sources.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.lookup_data_sources.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.lookup_data_sources.with_faulthandler` | Object | `` |
| `sql.worker.lookup_data_sources.write_int` | Function | `(value, stream)` |
| `sql.worker.lookup_data_sources.write_with_length` | Function | `(obj, stream)` |
| `sql.worker.plan_data_source_read.ArrowTableToRowsConversion` | Class | `(...)` |
| `sql.worker.plan_data_source_read.BinaryType` | Class | `(...)` |
| `sql.worker.plan_data_source_read.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.worker.plan_data_source_read.DataSourceReader` | Class | `(...)` |
| `sql.worker.plan_data_source_read.DataSourceStreamReader` | Class | `(...)` |
| `sql.worker.plan_data_source_read.InputPartition` | Class | `(value: Any)` |
| `sql.worker.plan_data_source_read.LocalDataToArrowConversion` | Class | `(...)` |
| `sql.worker.plan_data_source_read.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.worker.plan_data_source_read.PySparkRuntimeError` | Class | `(...)` |
| `sql.worker.plan_data_source_read.Row` | Class | `(...)` |
| `sql.worker.plan_data_source_read.SpecialLengths` | Class | `(...)` |
| `sql.worker.plan_data_source_read.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.worker.plan_data_source_read.auth_secret` | Object | `` |
| `sql.worker.plan_data_source_read.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `sql.worker.plan_data_source_read.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.plan_data_source_read.conn_info` | Object | `` |
| `sql.worker.plan_data_source_read.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.plan_data_source_read.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.plan_data_source_read.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.plan_data_source_read.pickleSer` | Object | `` |
| `sql.worker.plan_data_source_read.read_bool` | Function | `(stream)` |
| `sql.worker.plan_data_source_read.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `sql.worker.plan_data_source_read.read_int` | Function | `(stream)` |
| `sql.worker.plan_data_source_read.records_to_arrow_batches` | Function | `(output_iter: Union[Iterator[Tuple], Iterator[pa.RecordBatch]], max_arrow_batch_size: int, return_type: StructType, data_source: DataSource) -> Iterable[pa.RecordBatch]` |
| `sql.worker.plan_data_source_read.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.plan_data_source_read.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.plan_data_source_read.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.plan_data_source_read.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.plan_data_source_read.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.plan_data_source_read.to_arrow_schema` | Function | `(schema: StructType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.Schema` |
| `sql.worker.plan_data_source_read.utf8_deserializer` | Object | `` |
| `sql.worker.plan_data_source_read.with_faulthandler` | Object | `` |
| `sql.worker.plan_data_source_read.write_int` | Function | `(value, stream)` |
| `sql.worker.plan_data_source_read.write_read_func_and_partitions` | Function | `(outfile: IO, reader: Union[DataSourceReader, DataSourceStreamReader], data_source: DataSource, schema: StructType, max_arrow_batch_size: int, binary_as_bytes: bool) -> None` |
| `sql.worker.python_streaming_sink_runner.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.worker.python_streaming_sink_runner.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.worker.python_streaming_sink_runner.SpecialLengths` | Class | `(...)` |
| `sql.worker.python_streaming_sink_runner.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.worker.python_streaming_sink_runner.WriterCommitMessage` | Class | `(...)` |
| `sql.worker.python_streaming_sink_runner.auth_secret` | Object | `` |
| `sql.worker.python_streaming_sink_runner.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `sql.worker.python_streaming_sink_runner.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.python_streaming_sink_runner.conn_info` | Object | `` |
| `sql.worker.python_streaming_sink_runner.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.python_streaming_sink_runner.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.python_streaming_sink_runner.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.python_streaming_sink_runner.pickleSer` | Object | `` |
| `sql.worker.python_streaming_sink_runner.read_bool` | Function | `(stream)` |
| `sql.worker.python_streaming_sink_runner.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `sql.worker.python_streaming_sink_runner.read_int` | Function | `(stream)` |
| `sql.worker.python_streaming_sink_runner.read_long` | Function | `(stream)` |
| `sql.worker.python_streaming_sink_runner.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.python_streaming_sink_runner.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.python_streaming_sink_runner.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.python_streaming_sink_runner.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.python_streaming_sink_runner.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.python_streaming_sink_runner.utf8_deserializer` | Object | `` |
| `sql.worker.python_streaming_sink_runner.with_faulthandler` | Object | `` |
| `sql.worker.python_streaming_sink_runner.write_int` | Function | `(value, stream)` |
| `sql.worker.write_into_data_source.ArrowTableToRowsConversion` | Class | `(...)` |
| `sql.worker.write_into_data_source.BinaryType` | Class | `(...)` |
| `sql.worker.write_into_data_source.CaseInsensitiveDict` | Class | `(args: Any, kwargs: Any)` |
| `sql.worker.write_into_data_source.DataSource` | Class | `(options: Dict[str, str])` |
| `sql.worker.write_into_data_source.DataSourceArrowWriter` | Class | `(...)` |
| `sql.worker.write_into_data_source.DataSourceStreamArrowWriter` | Class | `(...)` |
| `sql.worker.write_into_data_source.DataSourceStreamWriter` | Class | `(...)` |
| `sql.worker.write_into_data_source.DataSourceWriter` | Class | `(...)` |
| `sql.worker.write_into_data_source.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `sql.worker.write_into_data_source.PySparkRuntimeError` | Class | `(...)` |
| `sql.worker.write_into_data_source.PySparkTypeError` | Class | `(...)` |
| `sql.worker.write_into_data_source.Row` | Class | `(...)` |
| `sql.worker.write_into_data_source.SpecialLengths` | Class | `(...)` |
| `sql.worker.write_into_data_source.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `sql.worker.write_into_data_source.WriterCommitMessage` | Class | `(...)` |
| `sql.worker.write_into_data_source.auth_secret` | Object | `` |
| `sql.worker.write_into_data_source.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `sql.worker.write_into_data_source.check_python_version` | Function | `(infile: IO) -> None` |
| `sql.worker.write_into_data_source.conn_info` | Object | `` |
| `sql.worker.write_into_data_source.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `sql.worker.write_into_data_source.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `sql.worker.write_into_data_source.main` | Function | `(infile: IO, outfile: IO) -> None` |
| `sql.worker.write_into_data_source.pickleSer` | Object | `` |
| `sql.worker.write_into_data_source.read_bool` | Function | `(stream)` |
| `sql.worker.write_into_data_source.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `sql.worker.write_into_data_source.read_int` | Function | `(stream)` |
| `sql.worker.write_into_data_source.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `sql.worker.write_into_data_source.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `sql.worker.write_into_data_source.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `sql.worker.write_into_data_source.setup_spark_files` | Function | `(infile: IO) -> None` |
| `sql.worker.write_into_data_source.start_faulthandler_periodic_traceback` | Object | `` |
| `sql.worker.write_into_data_source.utf8_deserializer` | Object | `` |
| `sql.worker.write_into_data_source.with_faulthandler` | Object | `` |
| `sql.worker.write_into_data_source.write_int` | Function | `(value, stream)` |
| `statcounter.StatCounter` | Class | `(values: Optional[Iterable[float]])` |
| `status` | Object | `` |
| `storagelevel.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `streaming.DStream` | Class | `(jdstream: JavaObject, ssc: StreamingContext, jrdd_deserializer: Serializer)` |
| `streaming.StreamingContext` | Class | `(sparkContext: SparkContext, batchDuration: Optional[int], jssc: Optional[JavaObject])` |
| `streaming.StreamingListener` | Class | `()` |
| `streaming.context.CloudPickleSerializer` | Class | `(...)` |
| `streaming.context.DStream` | Class | `(jdstream: JavaObject, ssc: StreamingContext, jrdd_deserializer: Serializer)` |
| `streaming.context.NoOpSerializer` | Class | `(...)` |
| `streaming.context.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `streaming.context.SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `streaming.context.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `streaming.context.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `streaming.context.StreamingContext` | Class | `(sparkContext: SparkContext, batchDuration: Optional[int], jssc: Optional[JavaObject])` |
| `streaming.context.StreamingListener` | Class | `()` |
| `streaming.context.T` | Object | `` |
| `streaming.context.TransformFunction` | Class | `(ctx, func, deserializers)` |
| `streaming.context.TransformFunctionSerializer` | Class | `(ctx, serializer, gateway)` |
| `streaming.context.UTF8Deserializer` | Class | `(use_unicode)` |
| `streaming.dstream.DStream` | Class | `(jdstream: JavaObject, ssc: StreamingContext, jrdd_deserializer: Serializer)` |
| `streaming.dstream.K` | Object | `` |
| `streaming.dstream.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `streaming.dstream.ResultIterable` | Class | `(data: SizedIterable[T])` |
| `streaming.dstream.S` | Object | `` |
| `streaming.dstream.Serializer` | Class | `(...)` |
| `streaming.dstream.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `streaming.dstream.StreamingContext` | Class | `(sparkContext: SparkContext, batchDuration: Optional[int], jssc: Optional[JavaObject])` |
| `streaming.dstream.T` | Object | `` |
| `streaming.dstream.T_co` | Object | `` |
| `streaming.dstream.TransformFunction` | Class | `(ctx, func, deserializers)` |
| `streaming.dstream.TransformedDStream` | Class | `(prev: DStream[T], func: Union[Callable[[RDD[T]], RDD[U]], Callable[[datetime, RDD[T]], RDD[U]]])` |
| `streaming.dstream.U` | Object | `` |
| `streaming.dstream.V` | Object | `` |
| `streaming.dstream.portable_hash` | Function | `(x: Hashable) -> int` |
| `streaming.dstream.rddToFileName` | Function | `(prefix, suffix, timestamp)` |
| `streaming.kinesis.DStream` | Class | `(jdstream: JavaObject, ssc: StreamingContext, jrdd_deserializer: Serializer)` |
| `streaming.kinesis.InitialPositionInStream` | Class | `(...)` |
| `streaming.kinesis.KinesisUtils` | Class | `(...)` |
| `streaming.kinesis.MetricsLevel` | Class | `(...)` |
| `streaming.kinesis.NoOpSerializer` | Class | `(...)` |
| `streaming.kinesis.StorageLevel` | Class | `(useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int)` |
| `streaming.kinesis.StreamingContext` | Class | `(sparkContext: SparkContext, batchDuration: Optional[int], jssc: Optional[JavaObject])` |
| `streaming.kinesis.T` | Object | `` |
| `streaming.kinesis.utf8_decoder` | Function | `(s: Optional[bytes]) -> Optional[str]` |
| `streaming.listener.StreamingListener` | Class | `()` |
| `streaming.util.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `streaming.util.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `streaming.util.TransformFunction` | Class | `(ctx, func, deserializers)` |
| `streaming.util.TransformFunctionSerializer` | Class | `(ctx, serializer, gateway)` |
| `streaming.util.rddToFileName` | Function | `(prefix, suffix, timestamp)` |
| `taskcontext.ALL_GATHER_FUNCTION` | Object | `` |
| `taskcontext.BARRIER_FUNCTION` | Object | `` |
| `taskcontext.BarrierTaskContext` | Class | `(...)` |
| `taskcontext.BarrierTaskInfo` | Class | `(address: str)` |
| `taskcontext.PySparkRuntimeError` | Class | `(...)` |
| `taskcontext.ResourceInformation` | Class | `(name: str, addresses: List[str])` |
| `taskcontext.TaskContext` | Class | `(...)` |
| `taskcontext.UTF8Deserializer` | Class | `(use_unicode)` |
| `taskcontext.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `taskcontext.read_int` | Function | `(stream)` |
| `taskcontext.write_int` | Function | `(value, stream)` |
| `taskcontext.write_with_length` | Function | `(obj, stream)` |
| `testing.assertDataFrameEqual` | Function | `(actual: Union[DataFrame, pandas.DataFrame, pyspark.pandas.DataFrame, List[Row]], expected: Union[DataFrame, pandas.DataFrame, pyspark.pandas.DataFrame, List[Row]], checkRowOrder: bool, rtol: float, atol: float, ignoreNullable: bool, ignoreColumnOrder: bool, ignoreColumnName: bool, ignoreColumnType: bool, maxErrors: Optional[int], showOnlyDiff: bool, includeDiffRows)` |
| `testing.assertSchemaEqual` | Function | `(actual: StructType, expected: StructType, ignoreNullable: bool, ignoreColumnOrder: bool, ignoreColumnName: bool)` |
| `testing.connectutils.DataFrame` | Class | `(plan: plan.LogicalPlan, session: SparkSession)` |
| `testing.connectutils.LogicalPlan` | Class | `(child: Optional[LogicalPlan], references: Optional[Sequence[LogicalPlan]])` |
| `testing.connectutils.LooseVersion` | Class | `(vstring: Optional[str])` |
| `testing.connectutils.MockRemoteSession` | Class | `()` |
| `testing.connectutils.PlanOnlyTestFixture` | Class | `(...)` |
| `testing.connectutils.PySparkErrorTestUtils` | Class | `(...)` |
| `testing.connectutils.PySparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `testing.connectutils.Range` | Class | `(start: int, end: int, step: int, num_partitions: Optional[int])` |
| `testing.connectutils.Read` | Class | `(table_name: str, options: Optional[Dict[str, str]], is_streaming: Optional[bool])` |
| `testing.connectutils.ReusedConnectTestCase` | Class | `(...)` |
| `testing.connectutils.ReusedMixedTestCase` | Class | `(...)` |
| `testing.connectutils.Row` | Class | `(...)` |
| `testing.connectutils.SQL` | Class | `(query: str, args: Optional[List[Column]], named_args: Optional[Dict[str, Column]], views: Optional[Sequence[SubqueryAlias]])` |
| `testing.connectutils.SQLTestUtils` | Class | `(...)` |
| `testing.connectutils.SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `testing.connectutils.SparkSession` | Class | `(connection: Union[str, DefaultChannelBuilder], userId: Optional[str], hook_factories: Optional[list[Callable[[SparkSession], Hook]]])` |
| `testing.connectutils.connect_requirement_message` | Object | `` |
| `testing.connectutils.googleapis_common_protos_requirement_message` | Object | `` |
| `testing.connectutils.graphviz_requirement_message` | Object | `` |
| `testing.connectutils.grpc_requirement_message` | Object | `` |
| `testing.connectutils.grpc_status_requirement_message` | Object | `` |
| `testing.connectutils.have_googleapis_common_protos` | Object | `` |
| `testing.connectutils.have_graphviz` | Object | `` |
| `testing.connectutils.have_grpc` | Object | `` |
| `testing.connectutils.have_grpc_status` | Object | `` |
| `testing.connectutils.have_pandas` | Object | `` |
| `testing.connectutils.is_remote_only` | Function | `() -> bool` |
| `testing.connectutils.pandas_requirement_message` | Object | `` |
| `testing.connectutils.pb2` | Object | `` |
| `testing.connectutils.pyarrow_requirement_message` | Object | `` |
| `testing.connectutils.should_test_connect` | Object | `` |
| `testing.connectutils.skip_if_server_version_is` | Function | `(cond: Callable[[LooseVersion], bool], reason: Optional[str]) -> Callable` |
| `testing.connectutils.skip_if_server_version_is_greater_than_or_equal_to` | Function | `(version: str, reason: Optional[str]) -> Callable` |
| `testing.mllibutils.MLlibTestCase` | Class | `(...)` |
| `testing.mllibutils.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `testing.mllibutils.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `testing.mlutils.ClassificationModel` | Class | `(...)` |
| `testing.mlutils.Classifier` | Class | `(...)` |
| `testing.mlutils.DataFrame` | Class | `(...)` |
| `testing.mlutils.DefaultParamsReadable` | Class | `(...)` |
| `testing.mlutils.DefaultParamsWritable` | Class | `(...)` |
| `testing.mlutils.DoubleType` | Class | `(...)` |
| `testing.mlutils.DummyEvaluator` | Class | `(...)` |
| `testing.mlutils.DummyLogisticRegression` | Class | `(featuresCol, labelCol, predictionCol, maxIter, regParam, rawPredictionCol)` |
| `testing.mlutils.DummyLogisticRegressionModel` | Class | `()` |
| `testing.mlutils.Estimator` | Class | `(...)` |
| `testing.mlutils.Evaluator` | Class | `(...)` |
| `testing.mlutils.HasFake` | Class | `()` |
| `testing.mlutils.HasMaxIter` | Class | `()` |
| `testing.mlutils.HasRegParam` | Class | `()` |
| `testing.mlutils.MockDataset` | Class | `()` |
| `testing.mlutils.MockEstimator` | Class | `()` |
| `testing.mlutils.MockModel` | Class | `(...)` |
| `testing.mlutils.MockTransformer` | Class | `()` |
| `testing.mlutils.MockUnaryTransformer` | Class | `(shiftVal)` |
| `testing.mlutils.Model` | Class | `(...)` |
| `testing.mlutils.Param` | Class | `(parent: Identifiable, name: str, doc: str, typeConverter: Optional[Callable[[Any], T]])` |
| `testing.mlutils.Params` | Class | `()` |
| `testing.mlutils.PySparkTestCase` | Class | `(...)` |
| `testing.mlutils.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `testing.mlutils.SparkSessionTestCase` | Class | `(...)` |
| `testing.mlutils.Transformer` | Class | `(...)` |
| `testing.mlutils.TypeConverters` | Class | `(...)` |
| `testing.mlutils.UnaryTransformer` | Class | `(...)` |
| `testing.mlutils.check_params` | Function | `(test_self, py_stage, check_params_exist)` |
| `testing.mlutils.keyword_only` | Function | `(func: _F) -> _F` |
| `testing.objects.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `testing.objects.DoubleType` | Class | `(...)` |
| `testing.objects.ExamplePoint` | Class | `(x, y)` |
| `testing.objects.ExamplePointUDT` | Class | `(...)` |
| `testing.objects.MyObject` | Class | `(key, value)` |
| `testing.objects.PythonOnlyPoint` | Class | `(...)` |
| `testing.objects.PythonOnlyUDT` | Class | `(...)` |
| `testing.objects.UTCOffsetTimezone` | Class | `(offset)` |
| `testing.objects.UserDefinedType` | Class | `(...)` |
| `testing.pandasutils.ComparisonTestBase` | Class | `(...)` |
| `testing.pandasutils.DataFrame` | Class | `(data, index, columns, dtype, copy)` |
| `testing.pandasutils.Index` | Class | `(...)` |
| `testing.pandasutils.PandasOnSparkTestCase` | Class | `(...)` |
| `testing.pandasutils.PandasOnSparkTestUtils` | Class | `(...)` |
| `testing.pandasutils.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `testing.pandasutils.ReusedSQLTestCase` | Class | `(...)` |
| `testing.pandasutils.SPARK_CONF_ARROW_ENABLED` | Object | `` |
| `testing.pandasutils.Series` | Class | `(data, index, dtype, name, copy, fastpath)` |
| `testing.pandasutils.TestUtils` | Class | `(...)` |
| `testing.pandasutils.assert_produces_warning` | Function | `(expected_warning, filter_level, check_stacklevel, raise_on_extra_warnings)` |
| `testing.pandasutils.compare_both` | Function | `(f, almost)` |
| `testing.pandasutils.is_ansi_mode_test` | Object | `` |
| `testing.pandasutils.ps` | Object | `` |
| `testing.pandasutils.require_minimum_pandas_version` | Function | `() -> None` |
| `testing.sqlutils.PySparkErrorTestUtils` | Class | `(...)` |
| `testing.sqlutils.ReusedPySparkTestCase` | Class | `(...)` |
| `testing.sqlutils.ReusedSQLTestCase` | Class | `(...)` |
| `testing.sqlutils.Row` | Class | `(...)` |
| `testing.sqlutils.SPARK_HOME` | Object | `` |
| `testing.sqlutils.SQLTestUtils` | Class | `(...)` |
| `testing.sqlutils.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `testing.sqlutils.have_pandas` | Object | `` |
| `testing.sqlutils.have_pyarrow` | Object | `` |
| `testing.sqlutils.pandas_requirement_message` | Object | `` |
| `testing.sqlutils.pyarrow_requirement_message` | Object | `` |
| `testing.sqlutils.require_test_compiled` | Function | `() -> None` |
| `testing.sqlutils.search_jar` | Function | `(project_relative_path, sbt_jar_name_prefix, mvn_jar_name_prefix)` |
| `testing.sqlutils.test_compiled` | Object | `` |
| `testing.sqlutils.test_not_compiled_message` | Object | `` |
| `testing.streamingutils.PySparkStreamingTestCase` | Class | `(...)` |
| `testing.streamingutils.RDD` | Class | `(jrdd: JavaObject, ctx: SparkContext, jrdd_deserializer: Serializer)` |
| `testing.streamingutils.SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `testing.streamingutils.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `testing.streamingutils.StreamingContext` | Class | `(sparkContext: SparkContext, batchDuration: Optional[int], jssc: Optional[JavaObject])` |
| `testing.streamingutils.existing_args` | Object | `` |
| `testing.streamingutils.jars_args` | Object | `` |
| `testing.streamingutils.kinesis_asl_assembly_jar` | Object | `` |
| `testing.streamingutils.kinesis_requirement_message` | Object | `` |
| `testing.streamingutils.kinesis_test_environ_var` | Object | `` |
| `testing.streamingutils.search_jar` | Function | `(project_relative_path, sbt_jar_name_prefix, mvn_jar_name_prefix)` |
| `testing.streamingutils.should_skip_kinesis_tests` | Object | `` |
| `testing.streamingutils.should_test_kinesis` | Object | `` |
| `testing.unittest_main` | Function | `(args, kwargs)` |
| `testing.utils.ByteArrayOutput` | Class | `()` |
| `testing.utils.DataFrame` | Class | `(...)` |
| `testing.utils.PySparkAssertionError` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], data: Optional[Iterable[Row]])` |
| `testing.utils.PySparkErrorTestUtils` | Class | `(...)` |
| `testing.utils.PySparkException` | Class | `(message: Optional[str], errorClass: Optional[str], messageParameters: Optional[Dict[str, str]], contexts: Optional[List[QueryContext]])` |
| `testing.utils.PySparkTestCase` | Class | `(...)` |
| `testing.utils.PySparkTypeError` | Class | `(...)` |
| `testing.utils.QueryContextType` | Class | `(...)` |
| `testing.utils.QuietTest` | Class | `(sc)` |
| `testing.utils.ReusedPySparkTestCase` | Class | `(...)` |
| `testing.utils.Row` | Class | `(...)` |
| `testing.utils.SparkConf` | Class | `(loadDefaults: bool, _jvm: Optional[JVMView], _jconf: Optional[JavaObject])` |
| `testing.utils.StructField` | Class | `(name: str, dataType: DataType, nullable: bool, metadata: Optional[Dict[str, Any]])` |
| `testing.utils.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `testing.utils.VariantVal` | Class | `(value: bytes, metadata: bytes)` |
| `testing.utils.ansi_mode_not_supported_message` | Object | `` |
| `testing.utils.assertDataFrameEqual` | Function | `(actual: Union[DataFrame, pandas.DataFrame, pyspark.pandas.DataFrame, List[Row]], expected: Union[DataFrame, pandas.DataFrame, pyspark.pandas.DataFrame, List[Row]], checkRowOrder: bool, rtol: float, atol: float, ignoreNullable: bool, ignoreColumnOrder: bool, ignoreColumnName: bool, ignoreColumnType: bool, maxErrors: Optional[int], showOnlyDiff: bool, includeDiffRows)` |
| `testing.utils.assertSchemaEqual` | Function | `(actual: StructType, expected: StructType, ignoreNullable: bool, ignoreColumnOrder: bool, ignoreColumnName: bool)` |
| `testing.utils.col` | Function | `(col: str) -> Column` |
| `testing.utils.connect_requirement_message` | Object | `` |
| `testing.utils.deepspeed_requirement_message` | Object | `` |
| `testing.utils.eventually` | Function | `(timeout, catch_assertions, catch_timeout)` |
| `testing.utils.flameprof_requirement_message` | Object | `` |
| `testing.utils.googleapis_common_protos_requirement_message` | Object | `` |
| `testing.utils.graphviz_requirement_message` | Object | `` |
| `testing.utils.grpc_requirement_message` | Object | `` |
| `testing.utils.grpc_status_requirement_message` | Object | `` |
| `testing.utils.have_deepspeed` | Object | `` |
| `testing.utils.have_flameprof` | Object | `` |
| `testing.utils.have_googleapis_common_protos` | Object | `` |
| `testing.utils.have_graphviz` | Object | `` |
| `testing.utils.have_grpc` | Object | `` |
| `testing.utils.have_grpc_status` | Object | `` |
| `testing.utils.have_jinja2` | Object | `` |
| `testing.utils.have_matplotlib` | Object | `` |
| `testing.utils.have_numpy` | Object | `` |
| `testing.utils.have_openpyxl` | Object | `` |
| `testing.utils.have_package` | Function | `(name: str) -> bool` |
| `testing.utils.have_pandas` | Object | `` |
| `testing.utils.have_plotly` | Object | `` |
| `testing.utils.have_pyarrow` | Object | `` |
| `testing.utils.have_scipy` | Object | `` |
| `testing.utils.have_sklearn` | Object | `` |
| `testing.utils.have_tabulate` | Object | `` |
| `testing.utils.have_torch` | Object | `` |
| `testing.utils.have_torcheval` | Object | `` |
| `testing.utils.have_yaml` | Object | `` |
| `testing.utils.is_ansi_mode_test` | Object | `` |
| `testing.utils.jinja2_requirement_message` | Object | `` |
| `testing.utils.matplotlib_requirement_message` | Object | `` |
| `testing.utils.numpy_requirement_message` | Object | `` |
| `testing.utils.openpyxl_requirement_message` | Object | `` |
| `testing.utils.pandas_requirement_message` | Object | `` |
| `testing.utils.plotly_requirement_message` | Object | `` |
| `testing.utils.pyarrow_requirement_message` | Object | `` |
| `testing.utils.read_int` | Function | `(b)` |
| `testing.utils.require_minimum_pandas_version` | Function | `() -> None` |
| `testing.utils.require_minimum_pyarrow_version` | Function | `() -> None` |
| `testing.utils.scipy_requirement_message` | Object | `` |
| `testing.utils.should_test_connect` | Object | `` |
| `testing.utils.sklearn_requirement_message` | Object | `` |
| `testing.utils.tabulate_requirement_message` | Object | `` |
| `testing.utils.timeout` | Function | `(timeout)` |
| `testing.utils.torch_requirement_message` | Object | `` |
| `testing.utils.torcheval_requirement_message` | Object | `` |
| `testing.utils.when` | Function | `(condition: Column, value: Any) -> Column` |
| `testing.utils.write_int` | Function | `(i)` |
| `testing.utils.yaml_requirement_message` | Object | `` |
| `traceback_utils.CallSite` | Object | `` |
| `traceback_utils.SCCallSiteSync` | Class | `(sc)` |
| `traceback_utils.first_spark_call` | Function | `()` |
| `util.ArrowCogroupedMapUDFType` | Object | `` |
| `util.ArrowGroupedAggUDFType` | Object | `` |
| `util.ArrowGroupedMapIterUDFType` | Object | `` |
| `util.ArrowGroupedMapUDFType` | Object | `` |
| `util.ArrowMapIterUDFType` | Object | `` |
| `util.ArrowScalarIterUDFType` | Object | `` |
| `util.ArrowScalarUDFType` | Object | `` |
| `util.ArrowWindowAggUDFType` | Object | `` |
| `util.GroupedMapUDFTransformWithStateInitStateType` | Object | `` |
| `util.GroupedMapUDFTransformWithStateType` | Object | `` |
| `util.InheritableThread` | Class | `(target: Callable, args: Any, session: Optional[SparkSession], kwargs: Any)` |
| `util.JVM_BYTE_MAX` | Object | `` |
| `util.JVM_BYTE_MIN` | Object | `` |
| `util.JVM_INT_MAX` | Object | `` |
| `util.JVM_INT_MIN` | Object | `` |
| `util.JVM_LONG_MAX` | Object | `` |
| `util.JVM_LONG_MIN` | Object | `` |
| `util.JVM_SHORT_MAX` | Object | `` |
| `util.JVM_SHORT_MIN` | Object | `` |
| `util.NonUDFType` | Object | `` |
| `util.PandasCogroupedMapUDFType` | Object | `` |
| `util.PandasGroupedAggUDFType` | Object | `` |
| `util.PandasGroupedMapIterUDFType` | Object | `` |
| `util.PandasGroupedMapUDFTransformWithStateInitStateType` | Object | `` |
| `util.PandasGroupedMapUDFTransformWithStateType` | Object | `` |
| `util.PandasGroupedMapUDFType` | Object | `` |
| `util.PandasGroupedMapUDFWithStateType` | Object | `` |
| `util.PandasMapIterUDFType` | Object | `` |
| `util.PandasScalarIterUDFType` | Object | `` |
| `util.PandasScalarUDFType` | Object | `` |
| `util.PandasWindowAggUDFType` | Object | `` |
| `util.PySparkRuntimeError` | Class | `(...)` |
| `util.PythonEvalType` | Class | `(...)` |
| `util.SQLArrowBatchedUDFType` | Object | `` |
| `util.SQLArrowTableUDFType` | Object | `` |
| `util.SQLArrowUDTFType` | Object | `` |
| `util.SQLBatchedUDFType` | Object | `` |
| `util.SQLTableUDFType` | Object | `` |
| `util.Serializer` | Class | `(...)` |
| `util.SparkContext` | Class | `(master: Optional[str], appName: Optional[str], sparkHome: Optional[str], pyFiles: Optional[List[str]], environment: Optional[Dict[str, Any]], batchSize: int, serializer: Serializer, conf: Optional[SparkConf], gateway: Optional[JavaGateway], jsc: Optional[JavaObject], profiler_cls: Type[BasicProfiler], udf_profiler_cls: Type[UDFBasicProfiler], memory_profiler_cls: Type[MemoryProfiler])` |
| `util.SparkSession` | Class | `(sparkContext: SparkContext, jsparkSession: Optional[JavaObject], options: Dict[str, Any])` |
| `util.SpecialLengths` | Class | `(...)` |
| `util.UTF8Deserializer` | Class | `(use_unicode)` |
| `util.VersionUtils` | Class | `(...)` |
| `util.default_api_mode` | Function | `() -> str` |
| `util.fail_on_stopiteration` | Function | `(f: Callable) -> Callable` |
| `util.globs` | Object | `` |
| `util.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `util.inheritable_thread_target` | Function | `(f: Optional[Union[Callable, SparkSession]]) -> Callable` |
| `util.is_remote_only` | Function | `() -> bool` |
| `util.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `util.print_exec` | Function | `(stream: TextIO) -> None` |
| `util.read_int` | Function | `(stream)` |
| `util.spark_connect_mode` | Function | `() -> str` |
| `util.start_faulthandler_periodic_traceback` | Object | `` |
| `util.try_simplify_traceback` | Function | `(tb: TracebackType) -> Optional[TracebackType]` |
| `util.walk_tb` | Function | `(tb: Optional[TracebackType]) -> Iterator[TracebackType]` |
| `util.with_faulthandler` | Object | `` |
| `util.write_int` | Function | `(value, stream)` |
| `util.write_with_length` | Function | `(obj, stream)` |
| `worker.ApplyInPandasWithStateSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, state_object_schema, arrow_max_records_per_batch, prefers_large_var_types, int_to_decimal_coercion_enabled)` |
| `worker.ArrayType` | Class | `(elementType: DataType, containsNull: bool)` |
| `worker.ArrowBatchUDFSerializer` | Class | `(timezone, safecheck, input_types, int_to_decimal_coercion_enabled, binary_as_bytes)` |
| `worker.ArrowStreamAggArrowUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_cast)` |
| `worker.ArrowStreamArrowUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_cast)` |
| `worker.ArrowStreamArrowUDTFSerializer` | Class | `(table_arg_offsets)` |
| `worker.ArrowStreamPandasUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, df_for_struct, struct_in_pandas, ndarray_as_list, arrow_cast, input_types, int_to_decimal_coercion_enabled)` |
| `worker.ArrowStreamPandasUDTFSerializer` | Class | `(timezone, safecheck, input_types, int_to_decimal_coercion_enabled)` |
| `worker.ArrowStreamUDFSerializer` | Class | `(...)` |
| `worker.ArrowStreamUDTFSerializer` | Class | `(...)` |
| `worker.ArrowTableToRowsConversion` | Class | `(...)` |
| `worker.BarrierTaskContext` | Class | `(...)` |
| `worker.BatchedSerializer` | Class | `(serializer, batchSize)` |
| `worker.BinaryType` | Class | `(...)` |
| `worker.CPickleSerializer` | Object | `` |
| `worker.CogroupArrowUDFSerializer` | Class | `(assign_cols_by_name)` |
| `worker.CogroupPandasUDFSerializer` | Class | `(...)` |
| `worker.DataType` | Class | `(...)` |
| `worker.GroupArrowUDFSerializer` | Class | `(assign_cols_by_name)` |
| `worker.GroupPandasIterUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, int_to_decimal_coercion_enabled)` |
| `worker.GroupPandasUDFSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, int_to_decimal_coercion_enabled)` |
| `worker.LocalDataToArrowConversion` | Class | `(...)` |
| `worker.MapType` | Class | `(keyType: DataType, valueType: DataType, valueContainsNull: bool)` |
| `worker.PySparkRuntimeError` | Class | `(...)` |
| `worker.PySparkTypeError` | Class | `(...)` |
| `worker.PySparkValueError` | Class | `(...)` |
| `worker.PythonEvalType` | Class | `(...)` |
| `worker.ResourceInformation` | Class | `(name: str, addresses: List[str])` |
| `worker.Row` | Class | `(...)` |
| `worker.SkipRestOfInputTableException` | Class | `(...)` |
| `worker.SpecialAccumulatorIds` | Class | `(...)` |
| `worker.SpecialLengths` | Class | `(...)` |
| `worker.StatefulProcessorApiClient` | Class | `(state_server_port: Union[int, str], key_schema: StructType, is_driver: bool)` |
| `worker.StringType` | Class | `(collation: str)` |
| `worker.StructType` | Class | `(fields: Optional[List[StructField]])` |
| `worker.TaskContext` | Class | `(...)` |
| `worker.TransformWithStateInPandasFuncMode` | Class | `(...)` |
| `worker.TransformWithStateInPandasInitStateSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_max_records_per_batch, arrow_max_bytes_per_batch, int_to_decimal_coercion_enabled)` |
| `worker.TransformWithStateInPandasSerializer` | Class | `(timezone, safecheck, assign_cols_by_name, arrow_max_records_per_batch, arrow_max_bytes_per_batch, int_to_decimal_coercion_enabled)` |
| `worker.TransformWithStateInPySparkRowInitStateSerializer` | Class | `(arrow_max_records_per_batch)` |
| `worker.TransformWithStateInPySparkRowSerializer` | Class | `(arrow_max_records_per_batch)` |
| `worker.assign_cols_by_name` | Function | `(runner_conf)` |
| `worker.auth_secret` | Object | `` |
| `worker.capture_outputs` | Function | `(context_provider: Callable[[], dict[str, str]]) -> Generator[None, None, None]` |
| `worker.chain` | Function | `(f, g)` |
| `worker.check_python_version` | Function | `(infile: IO) -> None` |
| `worker.conn_info` | Object | `` |
| `worker.fail_on_stopiteration` | Function | `(f: Callable) -> Callable` |
| `worker.handle_worker_exception` | Function | `(e: BaseException, outfile: IO, hide_traceback: Optional[bool]) -> None` |
| `worker.has_memory_profiler` | Object | `` |
| `worker.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `worker.main` | Function | `(infile, outfile)` |
| `worker.pickleSer` | Object | `` |
| `worker.read_bool` | Function | `(stream)` |
| `worker.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `worker.read_int` | Function | `(stream)` |
| `worker.read_long` | Function | `(stream)` |
| `worker.read_single_udf` | Function | `(pickleSer, infile, eval_type, runner_conf, udf_index, profiler)` |
| `worker.read_udfs` | Function | `(pickleSer, infile, eval_type)` |
| `worker.read_udtf` | Function | `(pickleSer, infile, eval_type)` |
| `worker.report_times` | Function | `(outfile, boot, init, finish)` |
| `worker.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `worker.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `worker.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `worker.setup_spark_files` | Function | `(infile: IO) -> None` |
| `worker.shuffle` | Object | `` |
| `worker.start_faulthandler_periodic_traceback` | Object | `` |
| `worker.to_arrow_type` | Function | `(dt: DataType, error_on_duplicated_field_names_in_struct: bool, timestamp_utc: bool, prefers_large_types: bool) -> pa.DataType` |
| `worker.use_large_var_types` | Function | `(runner_conf)` |
| `worker.use_legacy_pandas_udf_conversion` | Function | `(runner_conf)` |
| `worker.utf8_deserializer` | Object | `` |
| `worker.verify_arrow_batch` | Function | `(batch, assign_cols_by_name, expected_cols_and_types)` |
| `worker.verify_arrow_result` | Function | `(result, assign_cols_by_name, expected_cols_and_types)` |
| `worker.verify_arrow_table` | Function | `(table, assign_cols_by_name, expected_cols_and_types)` |
| `worker.verify_pandas_result` | Function | `(result, return_type, assign_cols_by_name, truncate_return_schema)` |
| `worker.with_faulthandler` | Object | `` |
| `worker.wrap_arrow_array_iter_udf` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_arrow_batch_iter_udf` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_arrow_batch_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_arrow_batch_udf_arrow` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_arrow_batch_udf_legacy` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_bounded_window_agg_arrow_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_bounded_window_agg_pandas_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_cogrouped_map_arrow_udf` | Function | `(f, return_type, argspec, runner_conf)` |
| `worker.wrap_cogrouped_map_pandas_udf` | Function | `(f, return_type, argspec, runner_conf)` |
| `worker.wrap_grouped_agg_arrow_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_grouped_agg_pandas_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_grouped_map_arrow_iter_udf` | Function | `(f, return_type, argspec, runner_conf)` |
| `worker.wrap_grouped_map_arrow_udf` | Function | `(f, return_type, argspec, runner_conf)` |
| `worker.wrap_grouped_map_pandas_iter_udf` | Function | `(f, return_type, argspec, runner_conf)` |
| `worker.wrap_grouped_map_pandas_udf` | Function | `(f, return_type, argspec, runner_conf)` |
| `worker.wrap_grouped_map_pandas_udf_with_state` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_grouped_transform_with_state_init_state_udf` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_grouped_transform_with_state_pandas_init_state_udf` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_grouped_transform_with_state_pandas_udf` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_grouped_transform_with_state_udf` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_kwargs_support` | Function | `(f, args_offsets, kwargs_offsets)` |
| `worker.wrap_memory_profiler` | Function | `(f, eval_type, result_id)` |
| `worker.wrap_pandas_batch_iter_udf` | Function | `(f, return_type, runner_conf)` |
| `worker.wrap_perf_profiler` | Function | `(f, eval_type, result_id)` |
| `worker.wrap_scalar_arrow_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_scalar_pandas_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type)` |
| `worker.wrap_unbounded_window_agg_arrow_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_unbounded_window_agg_pandas_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf)` |
| `worker.wrap_window_agg_arrow_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf, udf_index)` |
| `worker.wrap_window_agg_pandas_udf` | Function | `(f, args_offsets, kwargs_offsets, return_type, runner_conf, udf_index)` |
| `worker.write_int` | Function | `(value, stream)` |
| `worker.write_long` | Function | `(value, stream)` |
| `worker_util.CPickleSerializer` | Object | `` |
| `worker_util.FramedSerializer` | Class | `(...)` |
| `worker_util.PySparkRuntimeError` | Class | `(...)` |
| `worker_util.UTF8Deserializer` | Class | `(use_unicode)` |
| `worker_util.add_path` | Function | `(path: str) -> None` |
| `worker_util.check_python_version` | Function | `(infile: IO) -> None` |
| `worker_util.has_resource_module` | Object | `` |
| `worker_util.is_remote_only` | Function | `() -> bool` |
| `worker_util.local_connect_and_auth` | Function | `(conn_info: Optional[Union[str, int]], auth_secret: Optional[str]) -> Tuple` |
| `worker_util.pickleSer` | Object | `` |
| `worker_util.read_bool` | Function | `(stream)` |
| `worker_util.read_command` | Function | `(serializer: FramedSerializer, file: IO) -> Any` |
| `worker_util.read_int` | Function | `(stream)` |
| `worker_util.read_long` | Function | `(stream)` |
| `worker_util.send_accumulator_updates` | Function | `(outfile: IO) -> None` |
| `worker_util.setup_broadcasts` | Function | `(infile: IO) -> None` |
| `worker_util.setup_memory_limits` | Function | `(memory_limit_mb: int) -> None` |
| `worker_util.setup_spark_files` | Function | `(infile: IO) -> None` |
| `worker_util.utf8_deserializer` | Object | `` |
| `worker_util.write_int` | Function | `(value, stream)` |