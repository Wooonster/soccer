<template>
  <div class="score-panel">
    <div v-if="!results || results.length === 0" class="no-data">
      <el-empty description="暂无处理结果，请先处理视频" />
    </div>
    <div v-else class="results-container">
      <el-tabs type="border-card">
        <el-tab-pane 
          v-for="(result, index) in results" 
          :key="index"
          :label="getVideoName(result.filename || result.video_path)"
        >
          <div class="video-info">
            <h3>{{ getVideoName(result.filename || result.video_path) }}</h3>
            <p>检测到 {{ result.frame_count || (result.frame_idx ? result.frame_idx.length : 0) }} 个射门片段</p>
          </div>
          
          <el-table 
            :data="getTableData(result)" 
            stripe 
            class="result-table"
            height="calc(100% - 60px)"
          >
            <el-table-column prop="clipNumber" label="片段" min-width="60" align="center" />
            <el-table-column prop="frameId" label="帧号" min-width="80" align="center" />
            <el-table-column prop="timestamp" label="时间点" min-width="80" align="center">
              <template #default="scope">
                {{ formatTime(scope.row.timestamp) }}
              </template>
            </el-table-column>
            <el-table-column prop="confidence" label="置信度" min-width="120" align="center">
              <template #default="scope">
                <el-progress 
                  :percentage="Math.round(scope.row.confidence * 100)" 
                  :color="getConfidenceColor(scope.row.confidence)"
                />
              </template>
            </el-table-column>
            <el-table-column prop="clipPath" label="操作" min-width="120" align="center">
              <template #default="scope">
                <div class="preview-actions">
                  <el-button size="small" type="primary" @click="previewClip(scope.row.clipPath)">
                    预览
                  </el-button>
                  <el-button size="small" type="success" @click="downloadClip(scope.row.clipPath)">
                    下载
                  </el-button>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  results: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['preview-clip'])

// 获取视频文件名
function getVideoName(path) {
  if (!path) return '未知视频'
  return path.split('/').pop()
}

// 格式化时间戳为 mm:ss 格式
function formatTime(seconds) {
  if (seconds === undefined) return '--'
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

// 根据置信度获取进度条颜色
function getConfidenceColor(confidence) {
  if (confidence > 0.9) return '#67c23a'
  if (confidence > 0.75) return '#409eff'
  return '#e6a23c'
}

// 将结果数据转换为表格数据
function getTableData(result) {
  if (!result) return []
  
  console.log('Converting result to table data:', result)
  
  // 解析数据，兼容不同的字段名称
  const frameIds = result.frame_idx || []
  const timestamps = result.timestamps || result.timestamp || []
  const confidences = result.confidence || []
  const clipPaths = result.clip_paths || result.clip_path || []
  
  console.log('Extracted fields:', {
    frameIds: frameIds.length,
    timestamps: timestamps.length,
    confidences: confidences.length, 
    clipPaths: clipPaths.length
  })
  
  // 确保数组长度一致，取最小长度防止越界
  const minLength = Math.min(
    frameIds.length,
    timestamps.length,
    confidences.length,
    clipPaths.length
  )
  
  const tableData = []
  for (let i = 0; i < minLength; i++) {
    tableData.push({
      clipNumber: i + 1,
      frameId: frameIds[i],
      timestamp: timestamps[i],
      confidence: confidences[i],
      clipPath: clipPaths[i]
    })
  }
  
  console.log('Generated table data:', tableData)
  return tableData
}

// 预览视频片段
function previewClip(clipPath) {
  // 从路径中提取文件名
  const filename = clipPath.split('/').pop()
  emit('preview-clip', filename)
}

// 下载视频片段
function downloadClip(clipPath) {
  // 从路径中提取文件名
  const filename = clipPath.split('/').pop()
  
  // 创建下载链接
  const link = document.createElement('a')
  // link.href = `http://localhost:50001/api/download/${filename}`/
  link.href = `/api/download/${filename}`
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
</script>

<style scoped>
.score-panel {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.no-data {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.results-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

:deep(.el-tabs) {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

:deep(.el-tabs__header) {
  margin-bottom: 0;
}

:deep(.el-tabs__content) {
  flex: 1;
  overflow: auto;
  height: auto !important;
}

:deep(.el-tab-pane) {
  height: 100%;
  padding: 10px;
  overflow: auto;
  display: flex;
  flex-direction: column;
}

/* 表格样式 */
.result-table {
  flex: 1;
  width: 100%;
}

:deep(.el-table) {
  height: 100%;
  width: 100% !important;
}

:deep(.el-table__inner-wrapper) {
  height: 100%;
}

:deep(.el-table__body) {
  width: 100% !important;
}

:deep(.el-table__body td) {
  padding: 8px 0;
}

:deep(.el-table__body-wrapper) {
  overflow-y: auto;
}

:deep(.el-table__header-wrapper) {
  width: 100% !important;
}

:deep(.el-table__header) {
  width: 100% !important;
}

:deep(.el-table__empty-block) {
  height: 100%;
  width: 100% !important;
}

.video-info {
  margin-bottom: 15px;
}

.video-info h3 {
  margin: 0;
  margin-bottom: 5px;
}

.video-info p {
  margin: 0;
  color: #666;
}

.preview-actions {
  display: flex;
  gap: 10px;
  justify-content: center;
}

:deep(.el-progress) {
  width: 100%;
  margin: 0 auto;
}

:deep(.el-table .cell) {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 添加列内容样式 */
:deep(.el-table__row) {
  width: 100%;
}
</style>
