<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { statsApi } from '@/api'
import type { StatsOverview } from '@/api/types'
import { useEcharts } from '@/composables/useEcharts'
import { useUserStore } from '@/stores/user'

const router = useRouter()
const userStore = useUserStore()

const stats = ref<StatsOverview | null>(null)
const loading = ref(false)

// ---------------- 置信度分布 ----------------
const { el: bucketEl, render: renderBucket } = useEcharts(() => {
  const buckets = stats.value?.confidence_buckets ?? []
  return {
    tooltip: { trigger: 'item', formatter: '{b}: {c} 次（{d}%）' },
    legend: { bottom: 0 },
    series: [
      {
        type: 'pie',
        radius: ['42%', '68%'],
        center: ['50%', '46%'],
        data: buckets.map((b) => ({ name: b.label, value: b.count })),
        label: { formatter: '{b}\n{c} 次' },
        itemStyle: { borderColor: '#fff', borderWidth: 2 },
      },
    ],
  }
}, [stats])

// ---------------- 近 7 天趋势 ----------------
const { el: trendEl, render: renderTrend } = useEcharts(() => {
  const trend = stats.value?.daily_trend ?? []
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 44, right: 20, top: 24, bottom: 32 },
    xAxis: { type: 'category', data: trend.map((d) => d.date.slice(5)) },
    yAxis: { type: 'value', minInterval: 1, name: '识别次数' },
    series: [
      {
        type: 'line',
        smooth: true,
        areaStyle: { color: 'rgba(47,125,79,0.16)' },
        lineStyle: { color: '#2f7d4f', width: 2 },
        symbolSize: 7,
        data: trend.map((d) => d.count),
      },
    ],
  }
}, [stats])

// ---------------- 识别最多的类别 ----------------
const { el: topEl, render: renderTop } = useEcharts(() => {
  const tops = [...(stats.value?.top_classes ?? [])].reverse()
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 110, right: 46, top: 12, bottom: 24 },
    xAxis: { type: 'value', minInterval: 1 },
    yAxis: { type: 'category', data: tops.map((t) => t.name_cn), axisLabel: { fontSize: 12 } },
    series: [
      {
        type: 'bar',
        barWidth: 14,
        data: tops.map((t) => t.count),
        itemStyle: { color: '#67c23a', borderRadius: [0, 3, 3, 0] },
        label: { show: true, position: 'right', fontSize: 11 },
      },
    ],
  }
}, [stats])

async function load(): Promise<void> {
  loading.value = true
  try {
    stats.value = await statsApi.overview()
    renderBucket()
    renderTrend()
    renderTop()
  } catch {
    stats.value = null
  } finally {
    loading.value = false
  }
}

function formatTime(value: string | undefined): string {
  return value ? value.replace('T', ' ').slice(0, 19) : '—'
}

onMounted(load)
</script>

<template>
  <div class="page" v-loading="loading">
    <h2 class="page-title">个人中心</h2>
    <p class="page-subtitle">账号信息与我的识别统计（仅统计当前账号的数据）。</p>

    <el-row :gutter="18">
      <el-col :xs="24" :md="8">
        <el-card>
          <div class="avatar">🌸</div>
          <div class="uname">{{ userStore.user?.username }}</div>
          <el-descriptions :column="1" size="small" class="account-desc">
            <el-descriptions-item label="用户 ID">{{ userStore.user?.id ?? '—' }}</el-descriptions-item>
            <el-descriptions-item label="邮箱">{{ userStore.user?.email ?? '—' }}</el-descriptions-item>
            <el-descriptions-item label="注册时间">
              {{ formatTime(userStore.user?.created_at) }}
            </el-descriptions-item>
          </el-descriptions>
          <el-button class="go" type="primary" @click="router.push({ name: 'recognize' })">
            去识别花卉
          </el-button>
        </el-card>
      </el-col>

      <el-col :xs="24" :md="16">
        <el-row :gutter="14" class="stat-row">
          <el-col :span="6">
            <el-card class="stat"><div class="num">{{ stats?.total ?? 0 }}</div><div class="lbl">识别总次数</div></el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat"><div class="num">{{ stats?.today ?? 0 }}</div><div class="lbl">今日识别</div></el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat"><div class="num">{{ stats?.unique_classes ?? 0 }}</div><div class="lbl">识别过的花卉种类</div></el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat">
              <div class="num">{{ (stats?.avg_confidence ?? 0).toFixed(1) }}%</div>
              <div class="lbl">平均置信度</div>
            </el-card>
          </el-col>
        </el-row>

        <el-row :gutter="18">
          <el-col :xs="24" :lg="12">
            <el-card class="chart-card">
              <template #header><span>置信度分布</span></template>
              <div ref="bucketEl" class="chart"></div>
            </el-card>
          </el-col>
          <el-col :xs="24" :lg="12">
            <el-card class="chart-card">
              <template #header><span>近 7 天识别趋势</span></template>
              <div ref="trendEl" class="chart"></div>
            </el-card>
          </el-col>
        </el-row>

        <el-card class="chart-card">
          <template #header><span>识别最多的花卉（Top 10）</span></template>
          <div v-if="stats?.top_classes?.length" ref="topEl" class="chart tall"></div>
          <el-empty v-else description="还没有识别记录" :image-size="70" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.avatar {
  font-size: 46px;
  text-align: center;
}

.uname {
  margin: 6px 0 14px;
  text-align: center;
  font-size: 17px;
  font-weight: 600;
}

.account-desc {
  margin-bottom: 14px;
}

.go {
  width: 100%;
}

.stat-row {
  margin-bottom: 6px;
}

.stat {
  text-align: center;
}

.num {
  font-size: 24px;
  font-weight: 700;
  color: var(--brand);
}

.lbl {
  margin-top: 4px;
  color: var(--text-muted);
  font-size: 12px;
}

.chart-card {
  margin-top: 14px;
}

.chart {
  width: 100%;
  height: 250px;
}

.chart.tall {
  height: 300px;
}
</style>
