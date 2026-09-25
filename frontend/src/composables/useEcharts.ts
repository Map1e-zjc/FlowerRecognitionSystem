/**
 * ECharts 轻量封装：负责初始化、随容器尺寸自适应、组件卸载时释放。
 * 每个图表用量不大，不值得引入完整 wrapper 组件。
 */
import * as echarts from 'echarts'
import { onBeforeUnmount, onMounted, ref, watch, type Ref } from 'vue'

export function useEcharts(
  optionFactory: () => echarts.EChartsOption,
  deps: Ref<unknown>[] = [],
) {
  const el = ref<HTMLDivElement | null>(null)
  let chart: echarts.ECharts | null = null

  function render(): void {
    if (!el.value) return
    if (!chart) chart = echarts.init(el.value)
    chart.setOption(optionFactory(), true)
  }

  function resize(): void {
    chart?.resize()
  }

  onMounted(() => {
    render()
    window.addEventListener('resize', resize)
  })

  onBeforeUnmount(() => {
    window.removeEventListener('resize', resize)
    chart?.dispose()
    chart = null
  })

  if (deps.length) watch(deps, render, { deep: true })

  return { el, render, resize }
}
