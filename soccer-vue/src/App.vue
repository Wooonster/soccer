<script setup>
import { ref } from 'vue'
import VideoUpload from './components/VideoUpload.vue'
import ClipModeSelector from './components/ClipModeSelector.vue'
import VideoPreview from './components/VideoPreview.vue'
import ScorePanel from './components/ScorePanel.vue'
import { ElMessage } from 'element-plus'

const fileList = ref([])
const videos = ref([])
const clipMode = ref('goal')
const beforeSec = ref(3)
const afterSec = ref(5)
const previewVideo = ref(null)
const downloadUrl = ref('')
const processingResults = ref([])
const mergedVideoUrl = ref('')

function onUploaded(newVideos, newFileList) {
  console.log('onUploaded: ', newVideos, newFileList)
  videos.value = newVideos
  fileList.value = newFileList
}

async function onClipConfirm() {
  // 这个函数现在只负责设置参数
  console.log('Clip parameters set')
}

async function onProcessStart(resultFromEvent) {
  try {
    let result;
    
    // 如果是通过事件传来的结果，直接使用
    if (resultFromEvent) {
      console.log('Using results from event:', resultFromEvent);
      result = resultFromEvent;
    } else {
      // 否则发起请求获取结果
      const response = await fetch('http://localhost:50001/api/process-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });
      
      result = await response.json();
    }
    
    console.log('Processing results:', result);
    
    if (result.success) {
      // 更新处理结果
      if (result.results) {
        // 多个视频的结果
        console.log('Received multiple video results:', result.results);
        processingResults.value = result.results;
        
        // 保存合成视频的URL
        if (result.merged_video) {
          mergedVideoUrl.value = `http://localhost:50001/api/download/${result.merged_video}`
        }
      } else if (result.shot_result) {
        // 单个视频的结果
        console.log('Received single video result:', result.shot_result);
        processingResults.value = [result.shot_result];
        
        // 如果有合成视频
        if (result.merged_video) {
          mergedVideoUrl.value = `http://localhost:50001/api/download/${result.merged_video}`
        }
      }
      
      console.log('processingResults set to:', processingResults.value);
      
      // 如果有结果，自动预览第一个视频的第一个片段
      if (processingResults.value.length > 0) {
        const firstResult = processingResults.value[0];
        console.log('First result:', firstResult);
        
        const clipPaths = firstResult.clip_paths || firstResult.clip_path || [];
        console.log('Clip paths:', clipPaths);
        
        if (clipPaths && clipPaths.length > 0) {
          console.log('Preview first clip:', clipPaths[0]);
          previewClip(clipPaths[0]);
        } else {
          console.warn('No clip paths found in result');
        }
      } else {
        console.warn('No processing results available');
      }
    } else {
      console.error('Processing failed:', result.message);
    }
  } catch (error) {
    console.error('Error processing videos:', error);
  }
}

function previewClip(clipPath) {
  if (!clipPath) return
  
  // 如果是完整路径，提取文件名
  const filename = clipPath.includes('/') ? clipPath.split('/').pop() : clipPath
  
  // 设置预览视频的路径
  previewVideo.value = `http://localhost:50001/api/download/${filename}`
  // 同时更新下载链接
  downloadUrl.value = previewVideo.value
}

// 下载合成视频
function downloadMergedVideo() {
  if (!mergedVideoUrl.value) {
    ElMessage.warning('没有可下载的合成视频')
    return
  }
  
  // 创建一个临时链接元素并触发下载
  const link = document.createElement('a')
  link.href = mergedVideoUrl.value
  link.download = 'merged_video.mp4'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

// 清空所有数据
function clearAllData() {
  // 重置所有响应式状态
  fileList.value = []
  videos.value = []
  previewVideo.value = null
  downloadUrl.value = ''
  processingResults.value = []
  mergedVideoUrl.value = ''
  
  ElMessage.success('本地数据已清空')
}
</script>

<template>
  <el-container style="height: 100vh; width: 100vw; background: #f7f7f7; ">
    <el-row :gutter="20" style="height: 100%; width: 100%; margin-left: 10px; margin-right: 10px;;">
      <!-- 左侧 8 栅格 -->
      <el-col :span="8" style="display: flex; flex-direction: column; height: 100%; width: 100%;">
        <div style="display: flex; flex-direction: column; height: 100%; width: 100%;">
          <!-- upload -->
          <el-card style="width: 100%; height: 70%; margin-bottom: 20px; margin-right: 5px;">
            <VideoUpload :fileList="fileList" @uploaded="onUploaded" />
          </el-card>

          <!-- clip mode + beforeSec + afterSec -->
          <el-card style="width: 100%; height: 30%; margin-right: 5px;">
            <ClipModeSelector
              v-model:mode="clipMode"
              v-model:before="beforeSec"
              v-model:after="afterSec"
              @confirm="onClipConfirm"
              @process-start="onProcessStart"
              @download-merged="downloadMergedVideo"
              @clear-data="clearAllData"
            />
          </el-card>
        </div>
      </el-col>

      <!-- 右侧 16 栅格 -->
      <el-col :span="16" style="display: flex; flex-direction: column; height: 100%; width: 100%;">
        <!-- 视频预览：占比 55% -->
        <el-card
          class="preview-card"
          style="
            width: 100%;
            flex: 0 0 55%;
            margin-bottom: 20px;
            margin-left: 5px;
            display: flex;
            flex-direction: column;
          ">
          <VideoPreview :video="previewVideo" />
        </el-card>

        <!-- 评分面板：占比 45% -->
        <el-card
          class="score-card"
          style="
            width: 100%;
            flex: 1;
            margin-left: 5px;
            display: flex;
            flex-direction: column;
            padding: 0;
          ">
          <ScorePanel
            :results="processingResults"
            @preview-clip="previewClip"
          />
        </el-card>
      </el-col>
    </el-row>
  </el-container>
</template>

<style>
html, body, #app {
  height: 100%;
  margin: 0;
  padding: 0;
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  background: #f7f7f7;
}

/* 自定义卡片样式 */
.preview-card .el-card__body,
.score-card .el-card__body {
  height: 100%;
  padding: 0;
  display: flex;
  flex-direction: column;
}

/* 确保预览卡片内容正确填充空间 */
.preview-card .el-card__body {
  background-color: #000;
}

/* 评分面板卡片样式 */
.score-card .el-card__body {
  overflow: hidden;
}
</style>
