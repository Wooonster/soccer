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
  >
    <el-icon class="el-icon--upload"><upload-filled /></el-icon>
    <div class="el-upload__text">
      点击或拖拽上传视频
    </div>
    <template #tip>
      <div class="file-list-container">
        <div class="el-upload__tip">
          支持多文件上传，文件名会显示在下方（仅本地，不上传服务器）
        </div>
        <!-- 可滚动的文件列表 -->
        <div class="scrollable-file-list" style="min-height: 250px;">
          <div v-for="(file, index) in fileList" :key="index" class="file-item">
            <el-icon><document /></el-icon>
            <span class="file-name">{{ file.name }}</span>
            <el-icon class="delete-icon" @click="removeFile(index)"><close /></el-icon>
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

const fileList = ref([])

function handleChange(file, fileList_) {
  // fileList_ 已自动同步到 fileList
  // 你可以在这里做额外处理，比如限制数量
  // fileList.value = fileList_.slice(-3) // 只保留最新3个
}

function removeFile(index) {
  fileList.value.splice(index, 1)
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
}

.empty-tip {
  text-align: center;
  color: #909399;
  padding: 10px 0;
}

.delete-icon {
  position: absolute;
  right: 10px;
  color: #c0c4cc;
  cursor: pointer;
  transition: color 0.2s;
}

.delete-icon:hover {
  color: #f56c6c;
}
</style>
