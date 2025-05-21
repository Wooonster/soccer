<script setup>
import { ref } from 'vue'
import VideoUpload from './components/VideoUpload.vue'
import ClipModeSelector from './components/ClipModeSelector.vue'
import VideoPreview from './components/VideoPreview.vue'
import DownloadLink from './components/DownloadLink.vue'
import ScorePanel from './components/ScorePanel.vue'

const fileList = ref([])
const videos = ref([])
const clipMode = ref('goal')
const beforeSec = ref(3)
const afterSec = ref(5)
const previewVideo = ref(null)
const downloadUrl = ref('')
const score = ref(0.95)
const precision = ref(0.92)
const recall = ref(0.90)
const selectedFrame = ref(1234)

function onUploaded(newVideos, newFileList) {
  console.log('onUploaded: ', newVideos, newFileList)
  videos.value = newVideos
  fileList.value = newFileList
}

function onClipConfirm() {
  // TODO: 调用后端 API，处理视频并更新下方数据
  // previewVideo.value = 'xxx.mp4'
  // downloadUrl.value = 'xxx.mp4'
  // score.value = ...
  // precision.value = ...
  // recall.value = ...
  // selectedFrame.value = ...
}
</script>

<template>
  <el-container style="height: 100vh; width: 100vw; background: #f7f7f7;">
    <el-row :gutter="20" style="height: 100%; width: 100%;">
      <!-- 左侧 8 栅格 -->
      <el-col :span="8" style="height: 100%;">
        <div style="display: flex; flex-direction: column; height: 100%; width: 100%;">
          <!-- upload -->
          <el-card style="width: 100%; height: 55%; margin-bottom: 20px;">
            <VideoUpload :fileList="fileList" @uploaded="onUploaded" />
          </el-card>

          <!-- clip mode + beforeSec + afterSec -->
          <el-card style="width: 100%; height: 25%; margin-bottom: 20px;">
            <ClipModeSelector
              v-model:mode="clipMode"
              v-model:before="beforeSec"
              v-model:after="afterSec"
              @confirm="onClipConfirm"
            />
          </el-card>

          <!-- download -->
          <el-card style="width: 100%; height: 20%;">
            <DownloadLink :url="downloadUrl" />
          </el-card>
        </div>
      </el-col>

      <!-- 右侧 16 栅格 -->
      <el-col :span="16" style="display: flex; flex-direction: column; height: 100%; width: 100%;">
        <!-- 视频预览：占比 65% -->
        <el-card
          style="
            width: 100%;
            flex: 0 0 65%;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
          ">
          <VideoPreview :video="previewVideo" />
        </el-card>

        <!-- 评分面板：占比 35% -->
        <el-card
          style="
            width: 100%;
            flex: 1 1 auto;
            display: flex;
            align-items: center;
            justify-content: center;
          ">
          <ScorePanel
            :score="score"
            :precision="precision"
            :recall="recall"
            :frame="selectedFrame"
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
</style>
