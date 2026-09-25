/**
 * API 业务封装：按模块聚合，页面只调用这里的函数。
 */
import { api } from '@/api/client'
import type {
  ConfusionOut,
  CurvesOut,
  FlowerBrief,
  FlowerDetail,
  HistoryDetail,
  HistoryItem,
  ModelMetric,
  PageOut,
  PredictOut,
  StatsOverview,
  TokenOut,
  UserOut,
} from '@/api/types'

// ---------------- 认证 ----------------
export const authApi = {
  register(payload: { username: string; email: string; password: string }) {
    return api.post<UserOut>('/auth/register', payload)
  },
  login(payload: { username: string; password: string }) {
    return api.post<TokenOut>('/auth/login', payload)
  },
  me() {
    return api.get<UserOut>('/auth/me')
  },
}

// ---------------- 识别 ----------------
export const predictApi = {
  /** 上传图片识别；modelName 为空时后端使用默认（最优）模型 */
  predict(file: File, modelName?: string) {
    const form = new FormData()
    form.append('file', file)
    if (modelName) form.append('model_name', modelName)
    return api.upload<PredictOut>('/predict', form)
  },
}

// ---------------- 百科 ----------------
export const flowerApi = {
  list(params: { keyword?: string; page?: number; page_size?: number }) {
    return api.get<PageOut<FlowerBrief>>('/flowers', { params })
  },
  detail(classId: number) {
    return api.get<FlowerDetail>(`/flowers/${classId}`)
  },
}

// ---------------- 历史 ----------------
export const historyApi = {
  list(params: { page?: number; page_size?: number; class_id?: number }) {
    return api.get<PageOut<HistoryItem>>('/history', { params })
  },
  detail(recordId: number) {
    return api.get<HistoryDetail>(`/history/${recordId}`)
  },
  remove(recordId: number) {
    return api.delete<{ record_id: number; deleted: boolean }>(`/history/${recordId}`)
  },
}

// ---------------- 模型对比 ----------------
export const modelApi = {
  list() {
    return api.get<ModelMetric[]>('/models')
  },
  curves(name: string) {
    return api.get<CurvesOut>(`/models/${name}/curves`)
  },
  confusion(name: string) {
    return api.get<ConfusionOut>(`/models/${name}/confusion`)
  },
}

// ---------------- 统计 ----------------
export const statsApi = {
  overview() {
    return api.get<StatsOverview>('/stats/overview')
  },
}
