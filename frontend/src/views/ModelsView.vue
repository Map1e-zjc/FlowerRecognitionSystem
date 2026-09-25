<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import { modelApi } from '@/api'
import type { ConfusionOut, CurvesOut, ModelMetric } from '@/api/types'
import { useEcharts } from '@/composables/useEcharts'

const models = ref<ModelMetric[]>([])
const activeTab = ref('metrics')
const selected = ref<string>('')
const curves = ref<CurvesOut | null>(null)
const confusion = ref<ConfusionOut | null>(null)
const loading = ref(false)

const current = computed(() => models.value.find((m) => m.name === selected.value) ?? null)

// ---------------- 指标对比柱状图 ----------------
const { el: metricsEl, render: renderMetrics } = useEcharts(() => {
  const rows = [...models.value].sort((a, b) => b.top1 - a.top1)
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { data: ['Top-1', 'Top-5', 'Macro-F1'], bottom: 0 },
    grid: { left: 48, right: 24, top: 24, bottom: 48 },
    xAxis: {
      type: 'category',
      data: rows.map((m) => m.display_name),
      axisLabel: { fontSize: 12 },
    },
    yAxis: { type: 'value', min: 85, max: 100, axisLabel: { formatter: '{value}%' } },
    series: [
      { name: 'Top-1', type: 'bar', data: rows.map((m) => m.top1), barWidth: 18, itemStyle: { color: '#2f7d4f' } },
      { name: 'Top-5', type: 'bar', data: rows.map((m) => m.top5), barWidth: 18, itemStyle: { color: '#67c23a' } },
      { name: 'Macro-F1', type: 'bar', data: rows.map((m) => m.macro_f1), barWidth: 18, itemStyle: { color: '#e6a23c' } },
    ],
  }
}, [models])

// ---------------- 训练曲线 ----------------
const { el: curvesEl, render: renderCurves } = useEcharts(() => {
  const points = curves.value?.points ?? []
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['训练 Top-1', '验证 Top-1', '验证 Top-5'], bottom: 0 },
    grid: { left: 56, right: 24, top: 24, bottom: 48 },
    xAxis: { type: 'category', name: 'epoch', data: points.map((p) => p.epoch) },
    yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 24 }],
    series: [
      { name: '训练 Top-1', type: 'line', smooth: true, showSymbol: false, data: points.map((p) => p.train_top1), lineStyle: { color: '#909399' } },
      { name: '验证 Top-1', type: 'line', smooth: true, showSymbol: false, data: points.map((p) => p.val_top1), lineStyle: { color: '#2f7d4f', width: 2 } },
      { name: '验证 Top-5', type: 'line', smooth: true, showSymbol: false, data: points.map((p) => p.val_top5), lineStyle: { color: '#67c23a' } },
    ],
  }
}, [curves])

// ---------------- 混淆子矩阵热力图 ----------------
// 不渲染完整 102×102（10404 个格子会卡顿），只取「最易混淆对」涉及的类别
// 构成子矩阵，既快又直观看得出问题。
const subClasses = computed(() => {
  const c = confusion.value
  if (!c) return []
  const ids = new Set<number>()
  c.top_pairs.slice(0, 12).forEach((p) => {
    ids.add(p.true_id)
    ids.add(p.pred_id)
  })
  return [...ids].sort((a, b) => a - b)
})

const { el: heatEl, render: renderHeat } = useEcharts(() => {
  const c = confusion.value
  const ids = subClasses.value
  if (!c || ids.length === 0) return { series: [] }
  const data: [number, number, number][] = []
  let max = 0
  ids.forEach((ti, i) => {
    ids.forEach((pi, j) => {
      const v = c.matrix[ti]?.[pi] ?? 0
      if (v > 0) max = Math.max(max, v)
      data.push([j, i, v])
    })
  })
  const labels = ids.map((id) => c.class_names[id] ?? `class_${id}`)
  return {
    tooltip: {
      // 参数用 unknown 再窄化：ECharts 的回调参数是联合类型，
      // 直接声明成 { value: [number,number,number] } 会因逆变而类型不兼容。
      formatter: (params: unknown) => {
        const v = (params as { value: [number, number, number] }).value
        return `真实：${labels[v[1]]}<br/>预测：${labels[v[0]]}<br/>样本数：${v[2]}`
      },
    },
    grid: { left: 150, right: 60, top: 20, bottom: 80 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { rotate: 60, fontSize: 11, interval: 0 },
    },
    yAxis: { type: 'category', data: labels, axisLabel: { fontSize: 11 } },
    visualMap: { min: 0, max: Math.max(max, 1), calculable: true, orient: 'vertical', right: 6, top: 30 },
    series: [
      {
        type: 'heatmap',
        data,
        label: { show: ids.length <= 16, fontSize: 10 },
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.3)' } },
      },
    ],
  }
}, [confusion])

// ---------------- 最差类别柱状图 ----------------
const worstClasses = computed(() => {
  const c = confusion.value
  if (!c) return []
  return c.per_class_acc
    .map((acc, id) => ({ id, acc, name: c.class_names[id] ?? `class_${id}`, support: c.per_class_support[id] }))
    .filter((x) => x.support > 0)
    .sort((a, b) => a.acc - b.acc)
    .slice(0, 10)
    .reverse()
})

const { el: worstEl, render: renderWorst } = useEcharts(() => {
  const rows = worstClasses.value
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: unknown) => `${(v as number).toFixed(2)}%` },
    grid: { left: 130, right: 60, top: 12, bottom: 24 },
    xAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
    yAxis: { type: 'category', data: rows.map((r) => r.name), axisLabel: { fontSize: 12 } },
    series: [
      {
        type: 'bar',
        barWidth: 14,
        data: rows.map((r) => r.acc),
        itemStyle: {
          // 参数同样用 unknown 再窄化（ECharts 回调参数是联合类型）
          color: (p: unknown) => {
            const v = Number((p as { value: number }).value)
            return v >= 99 ? '#67c23a' : v >= 95 ? '#e6a23c' : '#f56c6c'
          },
          borderRadius: [0, 3, 3, 0],
        },
        label: {
          show: true,
          position: 'right',
          formatter: (p: unknown) => `${Number((p as { value: number }).value).toFixed(1)}%`,
          fontSize: 11,
        },
      },
    ],
  }
}, [confusion])

async function loadModels(): Promise<void> {
  loading.value = true
  try {
    models.value = await modelApi.list()
    if (!selected.value && models.value.length) {
      selected.value = models.value.find((m) => m.is_default)?.name ?? models.value[0].name
    }
    renderMetrics()
  } finally {
    loading.value = false
  }
}

async function loadDetail(): Promise<void> {
  if (!selected.value) return
  const [c, k] = await Promise.all([
    modelApi.curves(selected.value).catch(() => null),
    modelApi.confusion(selected.value).catch(() => null),
  ])
  curves.value = c
  confusion.value = k
  renderCurves()
  renderHeat()
  renderWorst()
}

watch(selected, loadDetail)
watch(activeTab, () => {
  // 切到对应 Tab 后容器才有尺寸，需要重算
  setTimeout(() => {
    renderCurves()
    renderHeat()
    renderWorst()
  }, 50)
})

onMounted(async () => {
  await loadModels()
  await loadDetail()
})
</script>

<template>
  <div class="page" v-loading="loading">
    <h2 class="page-title">模型对比与可视化</h2>
    <p class="page-subtitle">
      4 个主干网络在 Oxford Flowers-102 测试集（6149 张）上的实测结果。本页浏览无需登录。
    </p>

    <el-card class="chart-card">
      <template #header><span>指标对比</span></template>
      <div ref="metricsEl" class="chart tall"></div>
    </el-card>

    <el-card class="chart-card">
      <el-table :data="models" stripe size="small">
        <el-table-column label="模型" width="150">
          <template #default="{ row }">
            <b>{{ row.display_name }}</b>
            <el-tag v-if="row.is_default" size="small" type="success" class="badge">默认</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="pretrained" label="预训练来源" min-width="260" show-overflow-tooltip />
        <el-table-column label="Top-1" width="90" align="right" sortable :sort-method="(a: ModelMetric, b: ModelMetric) => a.top1 - b.top1">
          <template #default="{ row }">{{ row.top1.toFixed(2) }}%</template>
        </el-table-column>
        <el-table-column label="Top-5" width="90" align="right">
          <template #default="{ row }">{{ row.top5.toFixed(2) }}%</template>
        </el-table-column>
        <el-table-column label="Macro-F1" width="100" align="right">
          <template #default="{ row }">{{ row.macro_f1.toFixed(2) }}%</template>
        </el-table-column>
        <el-table-column label="参数" width="90" align="right">
          <template #default="{ row }">{{ row.params_m }} M</template>
        </el-table-column>
        <el-table-column label="推理(ms)" width="90" align="right">
          <template #default="{ row }">{{ row.infer_ms.toFixed(1) }}</template>
        </el-table-column>
        <el-table-column label="训练(min)" width="90" align="right">
          <template #default="{ row }">{{ row.train_minutes.toFixed(1) }}</template>
        </el-table-column>
        <el-table-column label="最佳epoch" width="90" align="right">
          <template #default="{ row }">{{ row.best_epoch }}/{{ row.epochs_run }}</template>
        </el-table-column>
      </el-table>
      <div class="note">
        结论：<b>预训练强度比网络架构更决定小样本细粒度任务的上限</b> —— ImageNet-22k / IN-21k
        预训练的两个模型测试集均 ≥99.4%，而仅 ImageNet-1k 预训练的 ResNet-50 为 94.26%。
      </div>
    </el-card>

    <el-card class="chart-card">
      <template #header>
        <div class="tab-header">
          <span>单个模型详情</span>
          <el-select v-model="selected" size="small" class="model-pick">
            <el-option
              v-for="m in models"
              :key="m.name"
              :label="`${m.display_name}（Top-1 ${m.top1}%）`"
              :value="m.name"
            />
          </el-select>
        </div>
      </template>

      <el-tabs v-model="activeTab">
        <el-tab-pane label="训练曲线" name="metrics">
          <div ref="curvesEl" class="chart tall"></div>
          <div class="note">
            验证集 Top-1 在 {{ current?.best_epoch }} epoch 达到最佳
            {{ current?.top1.toFixed(2) }}%，之后触发早停（共训练 {{ current?.epochs_run }} epoch）。
            训练集指标很早到 100%，说明 10 张/类的数据量下过拟合不可避免，靠增强与早停控制。
          </div>
        </el-tab-pane>

        <el-tab-pane label="混淆矩阵" name="confusion">
          <el-row :gutter="16">
            <el-col :xs="24" :lg="13">
              <div class="sub-title">最易混淆类别子矩阵（热力图可交互）</div>
              <div ref="heatEl" class="chart tall"></div>
            </el-col>
            <el-col :xs="24" :lg="11">
              <div class="sub-title">准确率最低的 10 个类别</div>
              <div ref="worstEl" class="chart tall"></div>
            </el-col>
          </el-row>

          <div class="sub-title">Top 混淆对（真实 → 误判）</div>
          <el-table :data="confusion?.top_pairs ?? []" size="small" stripe max-height="260">
            <el-table-column prop="true_name" label="真实类别" min-width="160" />
            <el-table-column prop="pred_name" label="误判为" min-width="160" />
            <el-table-column prop="count" label="次数" width="80" align="right" />
          </el-table>

          <div v-if="confusion?.figure" class="figure-block">
            <div class="sub-title">完整 102×102 混淆矩阵</div>
            <img :src="confusion.figure" class="figure" alt="混淆矩阵" loading="lazy" />
          </div>
        </el-tab-pane>

        <el-tab-pane label="错误案例" name="errors">
          <p class="note">
            每张图下方标注「真实类别」与「模型判断（置信度）」，均为**高置信度错分**样本
            —— 即模型"很自信地判错"的情况，最能反映数据本身的混淆边界。
          </p>
          <img
            v-if="current?.error_cases_figure"
            :src="current.error_cases_figure"
            class="figure"
            alt="错误案例"
            loading="lazy"
          />
          <el-empty v-else description="该模型暂无错误案例图" />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<style scoped>
.chart-card {
  margin-bottom: 18px;
}

.chart {
  width: 100%;
  height: 280px;
}

.chart.tall {
  height: 360px;
}

.tab-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.model-pick {
  width: 260px;
}

.sub-title {
  margin: 14px 0 8px;
  font-weight: 600;
  font-size: 13px;
}

.note {
  margin-top: 12px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.8;
}

.badge {
  margin-left: 6px;
}

.figure-block {
  margin-top: 10px;
}

.figure {
  width: 100%;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}
</style>
