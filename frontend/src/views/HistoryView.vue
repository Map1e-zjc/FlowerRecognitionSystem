<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { flowerApi, historyApi } from '@/api'
import type { FlowerBrief, HistoryDetail, HistoryItem } from '@/api/types'

const router = useRouter()

const items = ref<HistoryItem[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)
const filterClassId = ref<number | undefined>(undefined)
const loading = ref(false)
const flowers = ref<FlowerBrief[]>([])
const detail = ref<HistoryDetail | null>(null)
const detailVisible = ref(false)

const flowerOptions = computed(() =>
  flowers.value.map((f) => ({ label: `${f.name_cn}（${f.name_en}）`, value: f.class_id })),
)

async function load(): Promise<void> {
  loading.value = true
  try {
    const data = await historyApi.list({
      page: page.value,
      page_size: pageSize.value,
      class_id: filterClassId.value,
    })
    items.value = data.items
    total.value = data.total
  } catch {
    items.value = []
    total.value = 0
  } finally {
    loading.value = false
  }
}

async function loadFlowers(): Promise<void> {
  try {
    // 筛选项只需要中英文名，取前 102 条（全量）
    flowers.value = (await flowerApi.list({ page: 1, page_size: 100 })).items
    const rest = await flowerApi.list({ page: 2, page_size: 100 })
    flowers.value = [...flowers.value, ...rest.items]
  } catch {
    flowers.value = []
  }
}

function onPageChange(next: number): void {
  page.value = next
  void load()
}

function onFilterChange(): void {
  page.value = 1
  void load()
}

async function remove(row: HistoryItem): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `确定删除「${row.name_cn}」这条识别记录吗？删除后原图也会一并清理。`,
      '删除确认',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return // 用户取消
  }
  try {
    await historyApi.remove(row.id)
    ElMessage.success('已删除')
    if (items.value.length === 1 && page.value > 1) page.value -= 1
    await load()
  } catch {
    // 拦截器已提示
  }
}

async function openDetail(row: HistoryItem): Promise<void> {
  try {
    detail.value = await historyApi.detail(row.id)
    detailVisible.value = true
  } catch {
    // 拦截器已提示
  }
}

function formatTime(value: string): string {
  return value ? value.replace('T', ' ').slice(0, 19) : ''
}

onMounted(async () => {
  await Promise.all([load(), loadFlowers()])
})
</script>

<template>
  <div class="page">
    <h2 class="page-title">识别历史</h2>
    <p class="page-subtitle">仅显示当前账号的识别记录（账号之间数据完全隔离）。共 {{ total }} 条。</p>

    <div class="toolbar">
      <el-select
        v-model="filterClassId"
        placeholder="按花卉种类筛选"
        clearable
        filterable
        class="filter"
        @change="onFilterChange"
      >
        <el-option v-for="o in flowerOptions" :key="o.value" :label="o.label" :value="o.value" />
      </el-select>
      <el-button @click="onFilterChange">刷新</el-button>
    </div>

    <el-table v-loading="loading" :data="items" stripe @row-click="openDetail">
      <el-table-column label="原图" width="90">
        <template #default="{ row }">
          <img :src="row.image_url" class="mini" :alt="row.name_cn" loading="lazy" />
        </template>
      </el-table-column>
      <el-table-column label="识别结果" width="150">
        <template #default="{ row }">
          <div class="cell-cn">{{ row.name_cn }}</div>
          <div class="cell-en">{{ row.name_en }}</div>
        </template>
      </el-table-column>
      <el-table-column label="置信度" width="110" align="right">
        <template #default="{ row }">{{ (row.confidence * 100).toFixed(2) }}%</template>
      </el-table-column>
      <el-table-column prop="model_name" label="模型" width="140" />
      <el-table-column label="推理耗时" width="110" align="right">
        <template #default="{ row }">{{ row.latency_ms?.toFixed(1) }} ms</template>
      </el-table-column>
      <el-table-column label="识别时间" width="170">
        <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="170" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" @click.stop="openDetail(row)">查看 Top-5</el-button>
          <el-button
            link
            type="primary"
            @click.stop="router.push({ name: 'flower-detail', params: { classId: row.class_id } })"
          >
            百科
          </el-button>
          <el-button link type="danger" @click.stop="remove(row)">删除</el-button>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty description="还没有识别记录，去识别一张花卉照片吧">
          <el-button type="primary" @click="router.push({ name: 'recognize' })">去识别</el-button>
        </el-empty>
      </template>
    </el-table>

    <div class="pager">
      <el-pagination
        layout="prev, pager, next, total"
        :total="total"
        :current-page="page"
        :page-size="pageSize"
        background
        @current-change="onPageChange"
      />
    </div>

    <el-dialog v-model="detailVisible" title="识别详情（Top-5）" width="620px">
      <template v-if="detail">
        <div class="detail-top">
          <img :src="detail.image_url" class="detail-img" :alt="detail.name_cn" />
          <div class="detail-info">
            <div class="detail-cn">{{ detail.name_cn }}</div>
            <div class="cell-en">{{ detail.name_en }}</div>
            <div class="muted small">
              模型：{{ detail.model_name }} ｜ 耗时：{{ detail.latency_ms?.toFixed(1) }} ms
            </div>
            <div class="muted small">时间：{{ formatTime(detail.created_at) }}</div>
          </div>
        </div>
        <el-table :data="detail.top5" size="small" stripe class="detail-table">
          <el-table-column type="index" label="#" width="46" />
          <el-table-column prop="name_cn" label="中文名" width="120" />
          <el-table-column prop="name_en" label="英文名" show-overflow-tooltip />
          <el-table-column label="置信度" width="110" align="right">
            <template #default="{ row }">{{ (row.confidence * 100).toFixed(2) }}%</template>
          </el-table-column>
        </el-table>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 14px;
}

.filter {
  width: 280px;
}

.mini {
  width: 64px;
  height: 48px;
  object-fit: cover;
  border-radius: 4px;
  display: block;
}

.cell-cn {
  font-weight: 600;
}

.cell-en {
  color: var(--text-muted);
  font-size: 11px;
}

.pager {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}

.detail-top {
  display: flex;
  gap: 14px;
  margin-bottom: 12px;
}

.detail-img {
  width: 150px;
  height: 110px;
  object-fit: cover;
  border-radius: 6px;
}

.detail-cn {
  font-size: 18px;
  font-weight: 600;
  color: var(--brand);
}

.small {
  font-size: 12px;
}

.detail-info {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
</style>
