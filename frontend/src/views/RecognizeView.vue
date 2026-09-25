<script setup lang="ts">
import { ElMessage } from 'element-plus'
import type { UploadFile, UploadRawFile } from 'element-plus'
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { modelApi, predictApi } from '@/api'
import type { ModelMetric, PredictOut } from '@/api/types'
import { useEcharts } from '@/composables/useEcharts'

const router = useRouter()

const MAX_MB = 5
const ALLOWED = ['jpg', 'jpeg', 'png', 'webp']

const file = ref<File | null>(null)
const previewUrl = ref('')
const result = ref<PredictOut | null>(null)
const loading = ref(false)
const models = ref<ModelMetric[]>([])
const selectedModel = ref<string>('')

const top1 = computed(() => result.value?.predictions?.[0] ?? null)

// ---- Top-5 置信度条形图 ----
const { el: chartEl, render: renderChart } = useEcharts(() => {
  const preds = result.value?.predictions ?? []
  return {
    grid: { left: 96, right: 56, top: 12, bottom: 12 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      valueFormatter: (v: unknown) => `${((v as number) * 100).toFixed(2)}%`,
    },
    xAxis: {
      type: 'value',
      max: 1,
      axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: preds.map((p) => p.name_cn),
      axisLabel: { fontSize: 13 },
    },
    series: [
      {
        type: 'bar',
        data: preds.map((p) => p.confidence),
        barWidth: 18,
        itemStyle: { color: '#2f7d4f', borderRadius: [0, 4, 4, 0] },
        label: {
          show: true,
          position: 'right',
          // 参数用 unknown 再窄化：ECharts 回调参数是联合类型，直接声明窄类型会逆变不兼容
          formatter: (p: unknown) => `${(Number((p as { value: number }).value) * 100).toFixed(2)}%`,
          fontSize: 12,
        },
      },
    ],
  }
}, [result])

onMounted(async () => {
  try {
    models.value = await modelApi.list()
  } catch {
    // 模型列表拿不到不影响识别，后端会用默认模型
  }
})

/** 前端预校验，避免把明显不合规的文件发到后端 */
function beforeUpload(raw: UploadRawFile): boolean {
  const ext = raw.name.split('.').pop()?.toLowerCase() ?? ''
  if (!ALLOWED.includes(ext)) {
    ElMessage.error(`不支持的文件类型 .${ext}，仅支持 ${ALLOWED.join('、')}`)
    return false
  }
  if (raw.size > MAX_MB * 1024 * 1024) {
    ElMessage.error(`图片大小超过 ${MAX_MB} MB 限制`)
    return false
  }
  return true
}

function onFileChange(uploadFile: UploadFile): void {
  const raw = uploadFile.raw
  if (!raw) return
  file.value = raw
  result.value = null
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(raw)
}

function clearFile(): void {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
  file.value = null
  result.value = null
}

async function submit(): Promise<void> {
  if (!file.value) {
    ElMessage.warning('请先选择一张花卉图片')
    return
  }
  loading.value = true
  try {
    result.value = await predictApi.predict(file.value, selectedModel.value || undefined)
    renderChart()
    ElMessage.success('识别完成')
  } catch {
    // 拦截器已提示
  } finally {
    loading.value = false
  }
}

function openEncyclopedia(classId: number): void {
  void router.push({ name: 'flower-detail', params: { classId } })
}
</script>

<template>
  <div class="page">
    <h2 class="page-title">花卉识别</h2>
    <p class="page-subtitle">
      上传一张花卉照片，系统返回 Top-5 候选种类与置信度。支持 JPG / PNG / WEBP，单张不超过
      {{ MAX_MB }} MB。
    </p>

    <el-row :gutter="20">
      <!-- 左：上传 -->
      <el-col :xs="24" :md="11">
        <el-card>
          <template #header>
            <span>1. 选择图片</span>
          </template>

          <el-upload
            v-if="!previewUrl"
            drag
            :auto-upload="false"
            :show-file-list="false"
            accept=".jpg,.jpeg,.png,.webp"
            :before-upload="beforeUpload"
            :on-change="onFileChange"
          >
            <div class="upload-inner">
              <div class="upload-icon">🌸</div>
              <div>将图片拖到此处，或<em>点击选择</em></div>
              <div class="upload-hint">JPG / PNG / WEBP，≤ {{ MAX_MB }} MB</div>
            </div>
          </el-upload>

          <div v-else class="preview">
            <img :src="previewUrl" alt="待识别图片" />
            <div class="preview-name">{{ file?.name }}</div>
            <el-button size="small" @click="clearFile">重新选择</el-button>
          </div>

          <div class="model-select">
            <span class="muted">识别模型：</span>
            <el-select v-model="selectedModel" placeholder="默认使用最优模型" clearable size="small">
              <el-option
                v-for="m in models"
                :key="m.name"
                :label="`${m.display_name}（Top-1 ${m.top1}%）`"
                :value="m.name"
              />
            </el-select>
          </div>

          <el-button
            type="primary"
            class="submit"
            :loading="loading"
            :disabled="!file"
            @click="submit"
          >
            {{ loading ? '识别中…' : '开始识别' }}
          </el-button>
        </el-card>
      </el-col>

      <!-- 右：结果 -->
      <el-col :xs="24" :md="13">
        <el-card>
          <template #header>
            <span>2. 识别结果</span>
          </template>

          <el-empty v-if="!result" description="尚未识别，请先上传图片并点击「开始识别」" />

          <div v-else>
            <div class="top1">
              <div class="top1-name">{{ top1?.name_cn }}</div>
              <div class="top1-en">{{ top1?.name_en }}</div>
              <div class="top1-conf">{{ ((top1?.confidence ?? 0) * 100).toFixed(2) }}%</div>
            </div>

            <div class="meta">
              <el-tag size="small" type="success">{{ result.model_display_name }}</el-tag>
              <span class="muted">推理耗时 {{ result.latency_ms.toFixed(1) }} ms</span>
              <span class="muted">记录 #{{ result.record_id }}</span>
              <el-button
                v-if="top1"
                link
                type="primary"
                @click="openEncyclopedia(top1.class_id)"
              >
                查看该花卉百科 →
              </el-button>
            </div>

            <div ref="chartEl" class="chart"></div>

            <el-table :data="result.predictions" size="small" stripe>
              <el-table-column type="index" label="#" width="46" />
              <el-table-column prop="name_cn" label="中文名" width="110" />
              <el-table-column prop="name_en" label="英文名" show-overflow-tooltip />
              <el-table-column label="置信度" width="100" align="right">
                <template #default="{ row }">{{ (row.confidence * 100).toFixed(2) }}%</template>
              </el-table-column>
            </el-table>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.upload-inner {
  padding: 26px 10px;
  color: #6b7280;
}

.upload-icon {
  font-size: 44px;
  line-height: 1.2;
}

.upload-hint {
  margin-top: 6px;
  font-size: 12px;
  color: #9ca3af;
}

.preview {
  text-align: center;
}

.preview img {
  max-width: 100%;
  max-height: 300px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgb(0 0 0 / 10%);
}

.preview-name {
  margin: 8px 0;
  overflow: hidden;
  color: var(--text-muted);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-select {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 16px 0 12px;
  font-size: 13px;
}

.submit {
  width: 100%;
}

.top1 {
  padding: 14px 18px;
  border-radius: 10px;
  background: linear-gradient(135deg, #e8f5ee, #f7fbf9);
  text-align: center;
}

.top1-name {
  font-size: 26px;
  font-weight: 700;
  color: var(--brand);
}

.top1-en {
  color: var(--text-muted);
  font-size: 13px;
}

.top1-conf {
  margin-top: 4px;
  font-size: 18px;
  font-weight: 600;
}

.meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 12px 0 4px;
  font-size: 12px;
}

.chart {
  width: 100%;
  height: 200px;
}
</style>
