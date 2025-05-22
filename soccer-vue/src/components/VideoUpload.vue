<template>
  <el-upload
    v-model:file-list="fileList"
    class="upload-demo"
    drag
    multiple
    :auto-upload="false"
    :show-file-list="false"
    action="#"
    list-type="text"
    style="width: 100%;"
    :on-change="handleChange"
  >
    <el-icon class="el-icon--upload"><upload-filled /></el-icon>
    <div class="el-upload__text">
      点击或拖拽上传视频
    </div>
    <template #tip>
      <div class="file-list-container">
        <div class="el-upload__tip">
          支持多文件上传，文件名会显示在下方
        </div>
        <!-- 可滚动的文件列表 -->
        <div class="scrollable-file-list" style="min-height: 400px;">
          <div v-for="(file, index) in fileList" :key="index" class="file-item">
            <el-icon><document /></el-icon>
            <span class="file-name">{{ file.name }}</span>
            <div class="file-actions">
              <el-button 
                v-if="!file.uploaded" 
                size="small" 
                type="primary" 
                @click="uploadFile(file, index)"
                :loading="uploading === index"
              >
                上传
              </el-button>
              <el-tag v-if="file.uploaded" type="success">已上传</el-tag>
              <el-icon class="delete-icon" @click="removeFile(index)"><close /></el-icon>
            </div>
          </div>
          <div v-if="fileList.length === 0" class="empty-tip">暂无文件</div>
        </div>
      </div>
    </template>
  </el-upload>
</template>

<script setup>
import { ref } from 'vue'
import { UploadFilled, Document, Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const fileList = ref([])
const uploading = ref(null)

function handleChange(uploadFile) {
  // 将新上传的文件添加到列表中
  const file = {
    name: uploadFile.raw.name,
    size: uploadFile.raw.size,
    raw: uploadFile.raw,
    uploaded: false
  }
  
  // 检查是否已存在相同文件名的文件
  const exists = fileList.value.some(f => f.name === file.name)
  if (!exists) {
    fileList.value.push(file)
  } else {
    ElMessage.warning(`文件 ${file.name} 已存在`)
  }
}

function removeFile(index) {
  if (uploading.value === index) {
    ElMessage.warning('文件正在上传中，无法删除')
    return
  }
  fileList.value.splice(index, 1)
}

async function uploadFile(file, index) {
  if (file.uploaded) {
    ElMessage.info('该文件已上传')
    return
  }
  
  uploading.value = index
  
  const formData = new FormData()
  formData.append('file', file.raw)
  
  try {
    const response = await fetch('http://localhost:50001/api/upload-video', {
      method: 'POST',
      body: formData,
    })
    
    const result = await response.json()
    
    if (response.ok) {
      ElMessage.success('上传成功')
      fileList.value[index].uploaded = true
    } else {
      ElMessage.error(`上传失败: ${result.error || '未知错误'}`)
    }
  } catch (error) {
    ElMessage.error(`上传出错: ${error.message}`)
  } finally {
    uploading.value = null
  }
}
</script>

<style scoped>
.file-list-container {
  margin-top: 10px;
  width: 100%;
}

.scrollable-file-list {
  max-height: 150px;
  overflow-y: auto;
  margin-top: 10px;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  padding: 5px;
}

.file-item {
  display: flex;
  align-items: center;
  padding: 5px;
  border-bottom: 1px solid #f0f0f0;
  position: relative;
}

.file-item:last-child {
  border-bottom: none;
}

.file-name {
  margin-left: 5px;
  font-size: 14px;
  color: #606266;
  flex-grow: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.empty-tip {
  text-align: center;
  color: #909399;
  padding: 10px 0;
}

.delete-icon {
  color: #c0c4cc;
  cursor: pointer;
  transition: color 0.2s;
}

.delete-icon:hover {
  color: #f56c6c;
}
</style>
