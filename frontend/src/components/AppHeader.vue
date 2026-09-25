<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useUserStore } from '@/stores/user'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

const activePath = computed(() => {
  // 花卉详情页也让「花卉百科」保持高亮
  if (route.path.startsWith('/flowers')) return '/flowers'
  return route.path
})

const navItems = [
  { path: '/recognize', label: '花卉识别', requiresAuth: true },
  { path: '/flowers', label: '花卉百科', requiresAuth: false },
  { path: '/history', label: '识别历史', requiresAuth: true },
  { path: '/models', label: '模型对比', requiresAuth: false },
  { path: '/profile', label: '个人中心', requiresAuth: true },
]

function go(path: string, requiresAuth: boolean): void {
  if (requiresAuth && !userStore.isLoggedIn) {
    ElMessage.info('请先登录后再使用该功能')
    void router.push({ name: 'login', query: { redirect: path } })
    return
  }
  void router.push(path)
}

function handleLogout(): void {
  userStore.logout()
  ElMessage.success('已退出登录')
  void router.push({ name: 'login' })
}
</script>

<template>
  <header class="app-header">
    <div class="header-inner">
      <div class="brand" @click="go('/recognize', true)">
        <span class="brand-icon">🌸</span>
        <span class="brand-text">AI 花卉识别系统</span>
      </div>

      <nav class="nav">
        <a
          v-for="item in navItems"
          :key="item.path"
          class="nav-item"
          :class="{ active: activePath === item.path }"
          @click="go(item.path, item.requiresAuth)"
        >
          {{ item.label }}
        </a>
      </nav>

      <div class="account">
        <template v-if="userStore.isLoggedIn">
          <span class="username">{{ userStore.user?.username }}</span>
          <el-button link type="primary" @click="handleLogout">退出</el-button>
        </template>
        <template v-else>
          <el-button link type="primary" @click="router.push({ name: 'login' })">登录</el-button>
          <el-button link @click="router.push({ name: 'register' })">注册</el-button>
        </template>
      </div>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  background: #fff;
  border-bottom: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgb(0 0 0 / 4%);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-inner {
  display: flex;
  max-width: 1280px;
  margin: 0 auto;
  align-items: center;
  gap: 28px;
  padding: 0 24px;
  height: 56px;
}

.brand {
  display: flex;
  cursor: pointer;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  white-space: nowrap;
}

.brand-icon {
  font-size: 20px;
}

.nav {
  display: flex;
  flex: 1;
  gap: 4px;
}

.nav-item {
  cursor: pointer;
  border-radius: 6px;
  padding: 6px 12px;
  color: #374151;
  font-size: 14px;
  transition: all 0.15s;
}

.nav-item:hover {
  background: var(--brand-light);
  color: var(--brand);
}

.nav-item.active {
  background: var(--brand-light);
  color: var(--brand);
  font-weight: 600;
}

.account {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}

.username {
  max-width: 140px;
  overflow: hidden;
  color: #374151;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
