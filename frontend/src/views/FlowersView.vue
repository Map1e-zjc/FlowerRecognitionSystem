<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { flowerApi } from '@/api'
import type { FlowerBrief } from '@/api/types'

const router = useRouter()

const items = ref<FlowerBrief[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(12)
const keyword = ref('')
const loading = ref(false)

async function load(): Promise<void> {
  loading.value = true
  try {
    const data = await flowerApi.list({
      keyword: keyword.value.trim() || undefined,
      page: page.value,
      page_size: pageSize.value,
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

function search(): void {
  page.value = 1
  void load()
}

function reset(): void {
  keyword.value = ''
  search()
}

function onPageChange(next: number): void {
  page.value = next
  void load()
}

onMounted(load)
</script>

<template>
  <div class="page">
    <h2 class="page-title">花卉百科</h2>
    <p class="page-subtitle">
      共 {{ total }} 种花卉资料（含科属、花期、花色与养护要点）。本页浏览无需登录。
    </p>

    <div class="toolbar">
      <el-input
        v-model="keyword"
        placeholder="搜索中文名 / 英文名 / 科 / 属，例如：玫瑰、rose、蔷薇科"
        clearable
        class="search"
        @keyup.enter="search"
        @clear="search"
      />
      <el-button type="primary" @click="search">搜索</el-button>
      <el-button @click="reset">重置</el-button>
    </div>

    <div v-loading="loading">
      <el-empty v-if="!loading && items.length === 0" description="没有匹配的花卉，换个关键词试试" />

      <el-row v-else :gutter="16">
        <el-col v-for="f in items" :key="f.class_id" :xs="12" :sm="8" :md="6" :lg="4">
          <el-card
            class="flower-card"
            shadow="hover"
            :body-style="{ padding: '0' }"
            @click="router.push({ name: 'flower-detail', params: { classId: f.class_id } })"
          >
            <img
              v-if="f.sample_image"
              :src="f.sample_image"
              class="thumb"
              :alt="f.name_cn"
              loading="lazy"
            />
            <div v-else class="thumb thumb-empty">无图</div>
            <div class="flower-meta">
              <div class="flower-cn">{{ f.name_cn }}</div>
              <div class="flower-en">{{ f.name_en }}</div>
              <div class="flower-tags">
                <el-tag v-if="f.family" size="small" effect="plain">{{ f.family }}</el-tag>
                <el-tag v-if="f.bloom_season" size="small" type="success" effect="plain">
                  {{ f.bloom_season }}
                </el-tag>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

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
  </div>
</template>

<style scoped>
.toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 18px;
}

.search {
  max-width: 460px;
}

.flower-card {
  cursor: pointer;
  margin-bottom: 16px;
  overflow: hidden;
}

.thumb {
  width: 100%;
  height: 130px;
  object-fit: cover;
  display: block;
  background: #f3f4f6;
}

.thumb-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #b9bec7;
  font-size: 12px;
}

.flower-meta {
  padding: 8px 10px 10px;
}

.flower-cn {
  font-weight: 600;
  font-size: 14px;
}

.flower-en {
  overflow: hidden;
  color: var(--text-muted);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.flower-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}

.pager {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}
</style>
