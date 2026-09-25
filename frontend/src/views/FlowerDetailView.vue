<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { flowerApi } from '@/api'
import type { FlowerDetail } from '@/api/types'

const route = useRoute()
const router = useRouter()

const detail = ref<FlowerDetail | null>(null)
const loading = ref(false)
const failed = ref(false)

const classId = computed(() => Number(route.params.classId))

/** 空字段统一显示"暂无资料"，避免页面上出现空白行 */
function orDash(value: string | undefined): string {
  return value && value.trim() ? value : ''
}

const basicRows = computed(() => {
  if (!detail.value) return []
  const d = detail.value
  return [
    { label: '中文名', value: d.name_cn },
    { label: '英文名', value: d.name_en },
    { label: '类别', value: d.category_cn },
    { label: '科', value: d.family },
    { label: '属', value: d.genus },
    { label: '花期', value: d.bloom_season },
    { label: '花色', value: d.color },
  ]
})

const careRows = computed(() => {
  if (!detail.value) return []
  const d = detail.value
  return [
    { label: '光照', value: d.light },
    { label: '浇水', value: d.watering },
    { label: '土壤', value: d.soil },
    { label: '繁殖', value: d.propagation },
  ]
})

async function load(): Promise<void> {
  loading.value = true
  failed.value = false
  detail.value = null
  try {
    detail.value = await flowerApi.detail(classId.value)
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
}

onMounted(load)
watch(classId, load)
</script>

<template>
  <div class="page" v-loading="loading">
    <el-page-header content="花卉详情" @back="router.back()" />

    <el-result
      v-if="failed"
      icon="warning"
      title="未找到该花卉资料"
      :sub-title="`类别 ID：${classId}（有效范围 0—101）`"
    >
      <template #extra>
        <el-button type="primary" @click="router.push({ name: 'flowers' })">返回百科列表</el-button>
      </template>
    </el-result>

    <template v-else-if="detail">
      <el-row :gutter="20" class="detail-body">
        <el-col :xs="24" :md="8">
          <el-card :body-style="{ padding: '0' }">
            <img v-if="detail.sample_image" :src="detail.sample_image" class="cover" :alt="detail.name_cn" />
            <div v-else class="cover cover-empty">暂无图片</div>
          </el-card>
          <el-card class="side-card">
            <div class="side-title">识别提示</div>
            <p class="side-text">
              该类别的模型编号为 <b>{{ detail.class_id }}</b>。若识别结果指向本类，说明照片与训练样本
              的视觉特征高度接近。
            </p>
          </el-card>
        </el-col>

        <el-col :xs="24" :md="16">
          <el-card>
            <div class="name-row">
              <h2 class="cn-name">{{ detail.name_cn }}</h2>
              <span class="en-name">{{ detail.name_en }}</span>
            </div>

            <el-descriptions :column="2" border size="small" class="desc">
              <el-descriptions-item v-for="row in basicRows" :key="row.label" :label="row.label">
                <span v-if="orDash(row.value)">{{ row.value }}</span>
                <span v-else class="empty-field">暂无资料</span>
              </el-descriptions-item>
            </el-descriptions>

            <h4 class="section">简介</h4>
            <p v-if="orDash(detail.description)" class="para">{{ detail.description }}</p>
            <p v-else class="para empty-field">暂无资料</p>

            <h4 class="section">养护要点</h4>
            <el-descriptions :column="2" border size="small" class="desc">
              <el-descriptions-item v-for="row in careRows" :key="row.label" :label="row.label">
                <span v-if="orDash(row.value)">{{ row.value }}</span>
                <span v-else class="empty-field">暂无资料</span>
              </el-descriptions-item>
            </el-descriptions>
            <p v-if="orDash(detail.care_tips)" class="para tips">{{ detail.care_tips }}</p>
            <p v-else class="para empty-field">暂无资料</p>
          </el-card>
        </el-col>
      </el-row>
    </template>
  </div>
</template>

<style scoped>
.detail-body {
  margin-top: 18px;
}

.cover {
  width: 100%;
  height: 260px;
  object-fit: cover;
  display: block;
}

.cover-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #b9bec7;
  background: #f3f4f6;
}

.side-card {
  margin-top: 14px;
}

.side-title {
  margin-bottom: 6px;
  font-weight: 600;
  font-size: 14px;
}

.side-text {
  margin: 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.7;
}

.name-row {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

.cn-name {
  margin: 0;
  color: var(--brand);
  font-size: 24px;
}

.en-name {
  color: var(--text-muted);
  font-size: 14px;
}

.desc {
  margin: 16px 0 4px;
}

.section {
  margin: 18px 0 8px;
  font-size: 15px;
}

.para {
  margin: 6px 0 0;
  color: #374151;
  font-size: 13px;
  line-height: 1.8;
}

.tips {
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--brand-light);
}
</style>
