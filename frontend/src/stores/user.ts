/**
 * 用户状态（Pinia）：token 与当前用户信息，持久化到 localStorage。
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { authApi } from '@/api'
import { clearToken, getToken, setToken } from '@/api/client'
import type { UserOut } from '@/api/types'

const USER_KEY = 'flowers_user'

export const useUserStore = defineStore('user', () => {
  const token = ref<string>(getToken())
  const user = ref<UserOut | null>(readUser())

  const isLoggedIn = computed(() => Boolean(token.value))

  function readUser(): UserOut | null {
    const raw = localStorage.getItem(USER_KEY)
    if (!raw) return null
    try {
      return JSON.parse(raw) as UserOut
    } catch {
      return null
    }
  }

  function persist(nextUser: UserOut | null): void {
    user.value = nextUser
    if (nextUser) localStorage.setItem(USER_KEY, JSON.stringify(nextUser))
    else localStorage.removeItem(USER_KEY)
  }

  async function login(username: string, password: string): Promise<void> {
    const data = await authApi.login({ username, password })
    token.value = data.access_token
    setToken(data.access_token)
    persist(data.user)
  }

  async function register(payload: { username: string; email: string; password: string }): Promise<void> {
    await authApi.register(payload)
  }

  /** 用已存的 token 拉取最新用户信息（刷新页面后调用） */
  async function refresh(): Promise<boolean> {
    if (!token.value) return false
    try {
      persist(await authApi.me())
      return true
    } catch {
      logout()
      return false
    }
  }

  function logout(): void {
    token.value = ''
    clearToken()
    persist(null)
  }

  return { token, user, isLoggedIn, login, register, refresh, logout }
})
