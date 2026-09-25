/**
 * 路由与登录守卫。
 *
 * 按 docs/01 FR-04：百科与模型对比页**无需登录**；
 * 识别、历史、个人中心**必须登录**（meta.requiresAuth）。
 */
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

import { getToken } from '@/api/client'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/recognize' },
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: { title: '登录', publicPage: true },
  },
  {
    path: '/register',
    name: 'register',
    component: () => import('@/views/RegisterView.vue'),
    meta: { title: '注册', publicPage: true },
  },
  {
    path: '/recognize',
    name: 'recognize',
    component: () => import('@/views/RecognizeView.vue'),
    meta: { title: '花卉识别', requiresAuth: true },
  },
  {
    path: '/flowers',
    name: 'flowers',
    component: () => import('@/views/FlowersView.vue'),
    meta: { title: '花卉百科' },
  },
  {
    path: '/flowers/:classId',
    name: 'flower-detail',
    component: () => import('@/views/FlowerDetailView.vue'),
    meta: { title: '花卉详情' },
  },
  {
    path: '/history',
    name: 'history',
    component: () => import('@/views/HistoryView.vue'),
    meta: { title: '识别历史', requiresAuth: true },
  },
  {
    path: '/models',
    name: 'models',
    component: () => import('@/views/ModelsView.vue'),
    meta: { title: '模型对比' },
  },
  {
    path: '/profile',
    name: 'profile',
    component: () => import('@/views/ProfileView.vue'),
    meta: { title: '个人中心', requiresAuth: true },
  },
  { path: '/:pathMatch(.*)*', redirect: '/recognize' },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to) => {
  document.title = to.meta.title ? `${to.meta.title} · AI 花卉识别系统` : 'AI 花卉识别系统'
  if (to.meta.requiresAuth && !getToken()) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  return true
})

export default router
