<template>
  <div>
    <div style="font-size: 18px; font-weight: bold;">score: {{ score }}</div>
    <div>precision: {{ precision }}</div>
    <div>recall: {{ recall }}</div>
    <div>selected frame: {{ frame }}</div>
  </div>
</template>
<script setup>
const props = defineProps(['score', 'precision', 'recall', 'frame'])
</script>
