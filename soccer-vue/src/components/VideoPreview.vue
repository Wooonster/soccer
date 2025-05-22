<template>
  <div class="video-container">
    <video 
      v-if="video" 
      :src="video" 
      controls 
      class="video-player"
    />
    <div v-else class="empty-placeholder">视频预览区</div>
  </div>
</template>

<script setup>
const props = defineProps(['video'])
</script>

<style scoped>
.video-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background-color: #000;
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
</style>
