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
    <div class="selector-section button-group">
      <el-button 
        type="primary" 
        @click="onConfirm" 
        class="action-button"
        :loading="isSubmitting"
      >
        {{ isSubmitting ? '提交中...' : '设置参数' }}
      </el-button>
      <el-button 
        type="success" 
        @click="startProcessing" 
        class="action-button"
        :loading="isProcessing"
        :disabled="!paramsSet"
      >
        {{ isProcessing ? '处理中...' : '开始处理' }}
      </el-button>
    </div>
    
    <!-- 新增一行按钮 -->
    <div class="selector-section button-group">
      <el-button 
        type="primary" 
        @click="downloadMergedVideo" 
        class="action-button"
        :disabled="!hasMergedVideo"
      >
        <!-- <el-icon><download /></el-icon> -->
        下载合成视频
      </el-button>
      <el-button 
        type="danger" 
        @click="clearAllData" 
        class="action-button"
      >
        <!-- <el-icon><delete /></el-icon> -->
        清空所有数据
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
// import { Download, Delete } from '@element-plus/icons-vue'

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

const emit = defineEmits(['update:mode', 'update:before', 'update:after', 'confirm', 'process-start', 'download-merged', 'clear-data'])

// 本地状态
const isSubmitting = ref(false)
const isProcessing = ref(false)
const paramsSet = computed(() => localMode.value && localBefore.value >= 0 && localAfter.value >= 0)
const hasMergedVideo = ref(false) // 是否有合成视频可下载

// 当processStart事件发出并成功处理视频后，设置hasMergedVideo为true
// 此逻辑应在App.vue中添加，这里仅作为示例

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

async function startProcessing() {
  if (isProcessing.value) return

  if (!paramsSet.value) {
    ElMessage.warning('请先设置剪辑参数')
    return
  }

  isProcessing.value = true

  try {
    const response = await fetch('http://localhost:50001/api/process-video', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({})
    })

    const result = await response.json()
    
    if (response.ok) {
      ElMessage.success('处理成功')
      if (result.success && (result.results?.length > 0 || result.shot_result)) {
        hasMergedVideo.value = true // 更新状态，表示有合成视频可下载
      }
      emit('process-start', result)
    } else {
      ElMessage.error(`处理失败: ${result.error || '未知错误'}`)
    }
  } catch (error) {
    console.error('Clip processing error:', error)
    ElMessage.error(`处理出错: ${error.message || '网络错误'}`)
  } finally {
    isProcessing.value = false
  }
}

// 下载合成视频
function downloadMergedVideo() {
  if (!hasMergedVideo.value) {
    ElMessage.warning('没有可下载的合成视频')
    return
  }
  
  emit('download-merged')
  ElMessage.success('开始下载合成视频')
}

// 清除所有数据
function clearAllData() {
  ElMessageBox.confirm(
    '确定要清空所有数据吗？这将删除所有上传的视频和处理结果。',
    '确认清空',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(() => {
    // 发送API请求清除服务器上的文件
    fetch('http://localhost:50001/api/clear-all', {
      method: 'POST',
    }).then(response => {
      if (response.ok) {
        ElMessage.success('所有数据已清空')
        hasMergedVideo.value = false // 重置状态
        emit('clear-data') // 触发清空事件，让父组件重置所有状态
      } else {
        ElMessage.error('清空数据失败')
      }
    }).catch(error => {
      console.error('Clear data error:', error)
      ElMessage.error(`清空数据出错: ${error.message || '网络错误'}`)
    })
  }).catch(() => {
    // 用户取消操作
  })
}
</script>

<style scoped>
.clip-selector-container {
  padding: 10px;
}

.selector-section {
  margin-bottom: 15px;
}

.button-group {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}

.action-button {
  flex: 1;
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
</style>
