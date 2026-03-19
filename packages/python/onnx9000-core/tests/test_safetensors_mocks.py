def test_safetensors_mocks():
    from onnx9000.safetensors_mocks import (
        ray_mmap_ipc_deploy,
        grpc_serialize_tensor,
        celery_lazy_load_worker,
        linux_sendfile_weights,
        s3_boto3_range_request,
        azure_blob_range_request,
        gcp_chunked_load_memory,
        maximize_nvme_page_cache,
        load_tensor_parallel_slice,
        writev_vectorized_io,
        stream_arrays_sequentially,
        export_sharded_100gb_arrays,
        validate_rust_byte_parity,
        yield_stream_serialization,
        cross_platform_file_lock,
        validate_hub_etag,
        benchmark_7b_memory_usage,
        benchmark_1gb_layer_stream,
        profile_os_page_cache_hits,
        monitor_keep_alive_limits,
    )

    ray_mmap_ipc_deploy(None)
    grpc_serialize_tensor(None)
    celery_lazy_load_worker("")
    linux_sendfile_weights(None, None, 0, 0)
    s3_boto3_range_request(None, None, None)
    azure_blob_range_request(None, None)
    gcp_chunked_load_memory(None, None)
    maximize_nvme_page_cache(None)
    load_tensor_parallel_slice(None, None)
    writev_vectorized_io(None)
    stream_arrays_sequentially(None)
    export_sharded_100gb_arrays(None)
    validate_rust_byte_parity(None)
    list(yield_stream_serialization(None))
    cross_platform_file_lock(None)
    validate_hub_etag(None, None)
    benchmark_7b_memory_usage()
    benchmark_1gb_layer_stream()
    profile_os_page_cache_hits()
    monitor_keep_alive_limits()
