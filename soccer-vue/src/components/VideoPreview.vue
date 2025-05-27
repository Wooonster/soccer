<template>
  <div class="video-container">
    <video 
      v-if="video" 
      :src="video" 
      controls 
      class="video-player"
      @loadstart="onLoadStart"
      @loadedmetadata="onLoadedMetadata"
      @error="onError"
      @canplay="onCanPlay"
    />
    <div v-else class="empty-placeholder">视频预览区</div>
    <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps(['video'])
const errorMessage = ref('')

// 监听video prop变化
watch(() => props.video, (newVideo) => {
  console.log('VideoPreview: New video prop:', newVideo)
  errorMessage.value = '' // 清除之前的错误消息
})

function onLoadStart() {
  console.log('VideoPreview: Load start')
  errorMessage.value = ''
}

function onLoadedMetadata(event) {
  console.log('VideoPreview: Metadata loaded', event.target.duration)
}

function onError(event) {
  console.error('VideoPreview: Video load error', event)
  errorMessage.value = '视频加载失败，请检查文件是否存在'
}

function onCanPlay() {
  console.log('VideoPreview: Can play')
}
</script>

<style scoped>
.video-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  overflow: hidden;
  background-color: #000;
  position: relative;
}

.video-player {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.empty-placeholder {
  color: #aaa;
  font-size: 1.5rem;
  text-align: center;
}

.error-message {
  color: #f56c6c;
  font-size: 1rem;
  text-align: center;
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.7);
  padding: 10px;
  border-radius: 4px;
}
</style>
