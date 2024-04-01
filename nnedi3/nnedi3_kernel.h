#pragma once

#include <stdint.h>
#include <memory>
#include <deque>
#include <cuda_runtime_api.h>

class cudaEventPlanes {
protected:
    cudaEvent_t start;
    cudaEvent_t endY;
    cudaEvent_t endU;
    cudaEvent_t endV;
    cudaStream_t streamMain;
    cudaStream_t streamY;
    cudaStream_t streamU;
    cudaStream_t streamV;
public:
    cudaEventPlanes();
    ~cudaEventPlanes();
    void init();
    void startPlane(cudaStream_t sMain, cudaStream_t sY, cudaStream_t sU, cudaStream_t sV);
    void finPlane();
    bool planeYFin();
    bool planeUFin();
    bool planeVFin();
};

class CudaPlaneEventsPool {
protected:
    std::deque<std::unique_ptr<cudaEventPlanes>> events;
public:
    CudaPlaneEventsPool();
    ~CudaPlaneEventsPool();

    cudaEventPlanes *PlaneStreamStart(cudaStream_t sMain, cudaStream_t sY, cudaStream_t sU, cudaStream_t sV);
};

class cudaPlaneStreams {
    cudaStream_t stream;
    cudaStream_t streamY;
    cudaStream_t streamU;
    cudaStream_t streamV;
    CudaPlaneEventsPool eventPool;
public:
    cudaPlaneStreams();
    ~cudaPlaneStreams();
    void initStream(cudaStream_t stream_);
    cudaEventPlanes *CreateEventPlanes();
    void *GetDeviceStreamY();
    void *GetDeviceStreamU();
    void *GetDeviceStreamV();
    void *GetDeviceStreamPlane(int idx);
};

void GetWorkBytes(int width, int height, int& nnBytes, int& blockBytes);
void CopyPadCUDA(int pixelsize, uint8_t* ref, int refpitch, const uint8_t* src, int srcpitch, int width, int height, PNeoEnv env);
void BitBltCUDA(uint8_t* dst, int dstpitch, const uint8_t* src, int srcpitch, int width, int height, PNeoEnv env);
void PadRefAndCopyHalfCUDA(
    uint8_t *dst, const int dstpitch, uint8_t *ref, const int refpitch, const uint8_t *src, const int srcpitch, const int width, const int height, void *stream_, PNeoEnv env);
void EvalCUDA(int pixelsize, int bits_per_pixel,
  uint8_t* dst, int dstpitch, const uint8_t* ref, int refpitch, int width, int height,
  const int16_t* weights0, const int16_t* weights1, int weights1pitch,
  uint8_t* workNN, uint8_t* workBlock,
  int range_mode, int qual, int nns, int xdia, int ydia, void *stream_, PNeoEnv env);
