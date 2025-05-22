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
      <el-button 
        type="primary" 
        @click="onConfirm" 
        class="confirm-button"
        :loading="isSubmitting"
      >
        {{ isSubmitting ? '提交中...' : '确认' }}
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'

const props = defineProps({
  mode: {
    type: String,
    required: true
  },
  before: {
    type: Number,
    required: true
  },
  after: {
    type: Number,
    required: true
  }
})

const emit = defineEmits(['update:mode', 'update:before', 'update:after', 'confirm'])

// 本地状态
const isSubmitting = ref(false)

// 计算属性
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

async function onConfirm() {
  if (isSubmitting.value) return

  // 验证数据
  if (!localMode.value) {
    ElMessage.warning('请选择剪辑模式')
    return
  }
  if (localBefore.value < 0 || localAfter.value < 0) {
    ElMessage.warning('时间不能为负数')
    return
  }

  const data = {
    mode: localMode.value,
    before: localBefore.value,
    after: localAfter.value
  }

  isSubmitting.value = true

  try {
    const response = await fetch('http://localhost:50001/api/clip-mode', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })

    const result = await response.json()
    
    if (response.ok) {
      ElMessage.success('设置成功')
      emit('confirm')
    } else {
      ElMessage.error(`设置失败: ${result.error || '未知错误'}`)
    }
  } catch (error) {
    console.error('Clip mode submission error:', error)
    ElMessage.error(`提交出错: ${error.message || '网络错误'}`)
  } finally {
    isSubmitting.value = false
  }
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
