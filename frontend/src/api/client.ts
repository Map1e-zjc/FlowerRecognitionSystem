/**
 * Axios 实例与统一拦截器。
 *
 * 约定（docs/02 §7）：
 *  - baseURL 用相对路径 `/api/v1`，开发期由 Vite 代理到后端，避免硬编码域名与 CORS；
 *  - 请求拦截器自动注入 JWT；
 *  - 响应拦截器：业务码非 0 时抛出统一错误；401 自动清理登录态并跳登录页。
 */
import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios'
import { ElMessage } from 'element-plus'

import router from '@/router'

const TOKEN_KEY = 'flowers_token'

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

/** 业务错误：带上后端返回的业务码，便于调用方区分处理 */
export class ApiError extends Error {
  code: number
  constructor(code: number, message: string) {
    super(message)
    this.code = code
    this.name = 'ApiError'
  }
}

const http: AxiosInstance = axios.create({
  baseURL: '/api/v1',
  timeout: 60_000,
})

http.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers = config.headers ?? {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

http.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status
    const body = error?.response?.data
    const message: string =
      (body && typeof body.message === 'string' && body.message) ||
      (status === 401 ? '未登录或登录状态已失效，请重新登录' : '网络异常，请确认后端服务已启动')

    if (status === 401) {
      clearToken()
      const current = router.currentRoute.value
      if (current.name !== 'login') {
        ElMessage.warning(message)
        void router.push({ name: 'login', query: { redirect: current.fullPath } })
      }
    } else {
      ElMessage.error(message)
    }
    return Promise.reject(new ApiError(body?.code ?? status ?? -1, message))
  },
)

/**
 * 统一解包 `{ code, message, data }`，只把 `data` 交给调用方。
 * 业务码非 0 时弹错并抛出。
 */
async function unwrap<T>(promise: Promise<{ data: { code: number; message: string; data: T } }>): Promise<T> {
  const response = await promise
  const body = response.data
  if (body.code !== 0) {
    ElMessage.error(body.message || '请求失败')
    throw new ApiError(body.code, body.message)
  }
  return body.data
}

export const api = {
  get<T>(url: string, config?: AxiosRequestConfig) {
    return unwrap<T>(http.get(url, config))
  },
  post<T>(url: string, data?: unknown, config?: AxiosRequestConfig) {
    return unwrap<T>(http.post(url, data, config))
  },
  delete<T>(url: string, config?: AxiosRequestConfig) {
    return unwrap<T>(http.delete(url, config))
  },
  /** 上传（multipart/form-data） */
  upload<T>(url: string, form: FormData) {
    return unwrap<T>(http.post(url, form, { headers: { 'Content-Type': 'multipart/form-data' } }))
  },
}

export default http
