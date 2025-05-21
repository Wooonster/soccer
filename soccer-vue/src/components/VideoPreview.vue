<template>
  <div style="text-align: center; min-height: 300px;">
    <video v-if="video" :src="video" controls style="max-width: 100%; max-height: 400px;" />
    <div v-else style="color: #aaa; margin-top: 120px;">combined videos preview</div>
  </div>
</template>
<script setup>
const props = defineProps(['video'])
</script>
