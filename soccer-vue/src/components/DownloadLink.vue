<template>
  <div style="text-align: center; min-height: 60px;">
    <el-link v-if="url" :href="url" target="_blank" type="primary" :underline="false">
      下载合成视频
    </el-link>
    <div v-else style="color: #aaa;">combined video download link</div>
  </div>
</template>
<script setup>
const props = defineProps(['url'])
</script>
