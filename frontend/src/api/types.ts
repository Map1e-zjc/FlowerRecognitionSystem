/**
 * 后端 API 的类型定义。
 * 与 docs/02 §4 的接口契约、后端 backend/app/schemas/* 一一对应。
 */

/** 统一响应体：{ code, message, data } */
export interface Envelope<T> {
  code: number
  message: string
  data: T | null
}

export interface PageOut<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  pages: number
}

// ---------------- 认证 ----------------
export interface UserOut {
  id: number
  username: string
  email: string
  created_at: string
}

export interface TokenOut {
  access_token: string
  token_type: string
  expires_in: number
  user: UserOut
}

// ---------------- 识别 ----------------
export interface PredictionItem {
  class_id: number
  name_en: string
  name_cn: string
  confidence: number
}

export interface PredictOut {
  record_id: number
  model_name: string
  model_display_name: string
  model_pretrained: string
  latency_ms: number
  image_url: string
  predictions: PredictionItem[]
}

// ---------------- 花卉百科 ----------------
export interface FlowerBrief {
  class_id: number
  name_cn: string
  name_en: string
  category_cn: string
  family: string
  genus: string
  color: string
  bloom_season: string
  sample_image: string
}

export interface FlowerDetail extends FlowerBrief {
  category: string
  light: string
  watering: string
  soil: string
  propagation: string
  description: string
  care_tips: string
}

// ---------------- 识别历史 ----------------
export interface HistoryItem {
  id: number
  image_url: string
  class_id: number
  name_cn: string
  name_en: string
  confidence: number
  model_name: string
  latency_ms: number
  created_at: string
}

export interface HistoryDetail extends HistoryItem {
  top5: PredictionItem[]
}

// ---------------- 模型对比 ----------------
export interface ModelMetric {
  name: string
  tag: string
  display_name: string
  pretrained: string
  top1: number
  top5: number
  macro_f1: number
  params_m: number
  infer_ms: number
  infer_p95_ms: number
  train_minutes: number
  epochs_run: number
  best_epoch: number
  image_size: number
  num_samples: number
  is_default: boolean
  confusion_figure: string
  error_cases_figure: string
}

export interface CurvePoint {
  epoch: number
  train_loss: number | null
  train_top1: number | null
  val_loss: number | null
  val_top1: number | null
  val_top5: number | null
  epoch_seconds: number | null
}

export interface CurvesOut {
  name: string
  tag: string
  points: CurvePoint[]
}

export interface ConfusionPair {
  true_id: number
  true_name: string
  pred_id: number
  pred_name: string
  count: number
}

export interface ConfusionOut {
  name: string
  tag: string
  top1: number
  class_names: string[]
  matrix: number[][]
  per_class_acc: number[]
  per_class_support: number[]
  top_pairs: ConfusionPair[]
  figure: string
  num_samples: number
}

// ---------------- 统计 ----------------
export interface ConfidenceBucket {
  label: string
  count: number
}

export interface DailyPoint {
  date: string
  count: number
}

export interface TopClass {
  class_id: number
  name_cn: string
  name_en: string
  count: number
}

export interface StatsOverview {
  total: number
  today: number
  unique_classes: number
  avg_confidence: number
  confidence_buckets: ConfidenceBucket[]
  daily_trend: DailyPoint[]
  top_classes: TopClass[]
}

export interface HealthInfo {
  app: string
  env: string
  api_prefix: string
  database: string
  flowers: number
  model_metrics: number
  torch: string
  gpu_available: boolean
  gpu_name: string | null
  model_loaded: boolean
  model_name: string | null
  model_error: string | null
}
