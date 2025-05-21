<template>
  <div class="clip-selector-container">
    <!-- 选择模式：上部分 -->
    <div class="selector-section">
      <div class="section-title">剪辑模式：</div>
      <el-radio-group v-model="localMode" class="radio-group">
        <el-radio label="goal">进球</el-radio>
        <el-radio label="shot">打门</el-radio>
      </el-radio-group>
    </div>
    
    <!-- 前后秒数：中部分 -->
    <div class="selector-section">
      <div class="time-row">
        <span class="time-label">前置时间：</span>
        <el-input-number v-model="localBefore" :min="0" /> <span class="unit">秒</span>
      </div>
      <div class="time-row">
        <span class="time-label">后置时间：</span>
        <el-input-number v-model="localAfter" :min="0" /> <span class="unit">秒</span>
      </div>
    </div>
    
    <!-- 确认按钮：下部分 -->
    <div class="selector-section">
      <el-button type="primary" @click="onConfirm" class="confirm-button">确认</el-button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps(['mode', 'before', 'after'])
const emit = defineEmits(['update:mode', 'update:before', 'update:after', 'confirm'])

// 本地变量
const localMode = computed({
  get: () => props.mode,
  set: val => emit('update:mode', val)
})
const localBefore = computed({
  get: () => props.before,
  set: val => emit('update:before', val)
})
const localAfter = computed({
  get: () => props.after,
  set: val => emit('update:after', val)
})

function onConfirm() {
  console.log('onConfirm: ', localMode.value, localBefore.value, localAfter.value)
  emit('confirm')
}
</script>

<style scoped>
.clip-selector-container {
  padding: 10px;
}

.selector-section {
  margin-bottom: 15px;
}

.section-title {
  font-weight: bold;
  margin-bottom: 8px;
}

.radio-group {
  display: flex;
}

.time-row {
  margin-bottom: 10px;
  display: flex;
  align-items: center;
}

.time-label {
  width: 80px;
  text-align: right;
  margin-right: 10px;
}

.unit {
  margin-left: 5px;
}

.confirm-button {
  width: 100%;
}
</style>
