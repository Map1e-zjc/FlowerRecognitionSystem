"""API 路由聚合。

``api_router`` 汇总各业务模块的 router，由 ``app.main`` 一次性挂载。

设计说明
--------
这里**直接导入**各业务模块，不再用 ``try/except ImportError: pass`` 包裹。
原因：那种写法会让"某个模块导入失败"表现为**功能静默消失**（接口 404、OpenAPI 里
看不到该模块），排查成本极高。现在任何一个模块导入出错都会在启动时立刻抛出，
定位一目了然。

新增模块时在此追加一行 ``api_router.include_router(...)`` 即可。
"""

from fastapi import APIRouter

from app.api import auth, flowers, history, models, predict, stats

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["认证"])
api_router.include_router(predict.router, tags=["识别"])
api_router.include_router(flowers.router, prefix="/flowers", tags=["花卉百科"])
api_router.include_router(history.router, prefix="/history", tags=["识别历史"])
api_router.include_router(models.router, prefix="/models", tags=["模型对比"])
api_router.include_router(stats.router, prefix="/stats", tags=["统计"])

__all__ = ["api_router"]
